import torch

import torch_xla

from trainers.base_trainer import BaseTrainer
from models.ittt import ItttModel
from utils.loss_utils import lm_loss_fn
import utils.constants as constants


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


    @torch_xla.compile(full_graph=True)
    def first_chunk(self, chunks, ac_kwargs):

        with torch.autocast(**ac_kwargs):

            logits = self.model(
                chunks[0],
                logits_to_keep=slice(0, -1)
            )[0]
            loss = self.loss(chunks[0], logits)

        loss.backward()

        return loss


    @torch_xla.compile(full_graph=True)
    def looped_chunks(self, in_chunk, out_chunk, ac_kwargs):

        all_chunk = torch.cat([in_chunk, out_chunk], dim=-1)

        self.model.update_state()

        with torch.autocast(**ac_kwargs):

            logits = self.model(
                all_chunk,
                logits_to_keep=slice(in_chunk.shape[-1]-1, -1)
            )[0]
            loss = self.loss(
                all_chunk[:, in_chunk.shape[-1]-1:],
                logits
            )

        loss.backward()

        return loss


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

        # settings
        ac_kwargs = {
            "device_type": 'xla',
            "dtype": torch.bfloat16,
            "enabled": self.config.trainer.use_autocast,
        }

        input_ids: torch.LongTensor = batch["input_ids"]
        chunks = torch.split(
            input_ids, self.config.model.chunk_size,
            dim=-1
        )

        # first chunk
        total_loss = self.first_chunk(chunks, ac_kwargs)
        aux = {
            "lm_loss/chunk_00": total_loss,
        }

        # remaining chunks
        for i in range(1, len(chunks)):
            in_chunk = chunks[i-1]
            out_chunk = chunks[i]
            
            loss = self.looped_chunks(in_chunk, out_chunk, ac_kwargs)

            aux[f"lm_loss/chunk_{i:02d}"] = loss
            total_loss = total_loss + loss
        
        post_aux, grad_norm = self.post_forward()
        aux.update(post_aux)

        # finalize outputs
        final_loss = total_loss / len(chunks)
        aux["num_atoms"] = input_ids.numel()

        decades = {}
        for key, value in aux.items():

            if "chunk_" in key:
                if key.endswith("00"):
                    continue

                decade = int(key.split("_")[-1][0])

                if decade not in decades:
                    decades[decade] = []
                decades[decade].append(value)

        for decade, values in decades.items():
            aux[f"grouped_lm_loss/decade_{decade:02d}"] = torch.stack(values).mean()

        return final_loss, aux, grad_norm
    