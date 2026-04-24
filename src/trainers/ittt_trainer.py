import torch
import torch.nn.functional as F

import torch_xla

from trainers.base_trainer import BaseTrainer
from models.ittt import ItttModel
from utils.loss_utils import lm_loss_fn
import utils.constants as constants
from utils.logging_utils import master_print


class ItttTrainer(BaseTrainer):
    
    model: ItttModel


    def post_init(self):
        self.model.model.embed_tokens.weight.no_muon = True
        try:
            self.model.lm_head.weight.no_muon = True
        except:
            # ShardedModule
            self.model.lm_head._orig_mod.weight.no_muon = True

        self.model.init_state(
            self.global_batch_size, self.device
        )


    def loss(self, labels, logits):
        return lm_loss_fn(
            logits, labels,
            ignore_index=self.model.config.pad_token_id,
            shift_logits=False,
        )

    
    def nepa_loss(self, pred, target):
        return F.mse_loss(pred, target.detach())


    @torch_xla.compile(full_graph=True)
    def first_chunk(self, chunk):

        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):

            logits, _, nepa_pred, nepa_target = self.model(
                chunk,
                logits_to_keep=slice(0, -1)
            )
            lm_loss = self.loss(chunk, logits)

            nepa_loss = self.nepa_loss(nepa_pred[:, :-1], nepa_target[:, :-1])
            loss = lm_loss + self.config.trainer.nepa_loss_weight * nepa_loss

        loss.backward()

        return lm_loss, nepa_loss


    @torch_xla.compile(full_graph=True)
    def looped_chunks(self, in_chunk, out_chunk):

        all_chunk = torch.cat([in_chunk, out_chunk], dim=-1)

        self.model.update_state()

        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):

            logits, _, nepa_pred, nepa_target = self.model(
                all_chunk,
                logits_to_keep=slice(in_chunk.shape[-1]-1, -1)
            )
            lm_loss = self.loss(
                all_chunk[:, in_chunk.shape[-1]-1:],
                logits
            )

            nepa_loss = self.nepa_loss(
                nepa_pred[:, in_chunk.shape[-1]-1:-1],
                nepa_target[:, in_chunk.shape[-1]-1:-1]
            )

            loss = lm_loss + self.config.trainer.nepa_loss_weight * nepa_loss

        loss.backward()

        return lm_loss, nepa_loss


    @torch_xla.compile(full_graph=True)
    def post_forward(self):

        # clear state
        self.model.empty_state()

        # regular optimization step
        grad_norm = self.clip_gradients()

        aux = self.optimizers['main'].step()       
        self.model.zero_grad(set_to_none=False)

        aux['lr'] = self.lr_schedulers['main'].get_last_lr()[0]
        self.lr_schedulers['main'].step()

        return aux, grad_norm


    def train_step(self, batch):

        input_ids: torch.LongTensor = batch["input_ids"]
        chunks = torch.split(
            input_ids, self.config.model.chunk_size,
            dim=-1
        )

        # first chunk
        total_loss, total_nepa_loss = self.first_chunk(chunks[0])
        aux = {
            "lm_loss/chunk_00": total_loss,
            "nepa_loss/chunk_00": total_nepa_loss,
        }
        torch_xla.sync()
        master_print("Chunk 00 completed.")

        # remaining chunks
        for i in range(1, len(chunks)):
            in_chunk = chunks[i-1]
            out_chunk = chunks[i]
            
            loss, nepa_loss = self.looped_chunks(in_chunk, out_chunk)

            aux[f"lm_loss/chunk_{i:02d}"] = loss
            aux[f"nepa_loss/chunk_{i:02d}"] = nepa_loss
            total_loss = total_loss + loss
            total_nepa_loss = total_nepa_loss + nepa_loss
            torch_xla.sync()
            master_print(f"Chunk {i:02d} completed.")
        
        post_aux, grad_norm = self.post_forward()
        aux.update(post_aux)

        # finalize outputs
        final_loss = total_loss / len(chunks)
        final_nepa_loss = total_nepa_loss / len(chunks)
        aux["num_atoms"] = input_ids.numel()
        aux["total_nepa_loss"] = final_nepa_loss

        decades = {}
        nepa_decades = {}
        for key, value in aux.items():

            if "chunk_" in key and "lm_loss" in key:
                if key.endswith("00"):
                    continue

                decade = int(key.split("_")[-1][0])

                if decade not in decades:
                    decades[decade] = []
                decades[decade].append(value)

            elif "chunk_" in key and "nepa_loss" in key:
                if key.endswith("00"):
                    continue

                decade = int(key.split("_")[-1][0])

                if decade not in nepa_decades:
                    nepa_decades[decade] = []
                nepa_decades[decade].append(value)

        for decade, values in decades.items():
            aux[f"grouped_lm_loss/decade_{decade:02d}"] = torch.stack(values).mean()
        for decade, values in nepa_decades.items():
            aux[f"grouped_nepa_loss/decade_{decade:02d}"] = torch.stack(values).mean()

        return final_loss, aux, grad_norm
    