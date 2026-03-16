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
        with torch.autocast(**ac_kwargs):

            logits = self.model(
                chunks[0],
                logits_to_keep=slice(0, -1)
            )[0]
            loss = self.loss(chunks[0], logits)

        loss.backward()

        aux = {
            "lm_loss/chunk_00": loss,
        }
        total_loss = loss

        # remaining chunks
        for i in range(1, len(chunks)):
            in_chunk = chunks[i-1]
            out_chunk = chunks[i]
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

            aux[f"lm_loss/chunk_{i:02d}"] = loss
            total_loss = total_loss + loss
        
        # clear state
        self.model.empty_state()

        # regular optimization step
        grad_norm = self.clip_gradients()

        opt_aux = self.optimizers['main'].step()
        aux.update(opt_aux)        
        self.model.zero_grad(set_to_none=False)

        aux['lr'] = self.lr_schedulers['main'].get_last_lr()[0]
        self.lr_schedulers['main'].step()

        # finalize outputs
        final_loss = total_loss / len(chunks)
        aux["num_atoms"] = input_ids.numel()

        decades = {}
        for key, value in aux.items():

            if "chunk_" in key:
                if key.endswith("00"):
                    continue

                decade = key.split("_")[-1][0]

                if decade not in decades:
                    decades[decade] = []
                decades[decade].append(value)

        for decade, values in decades.items():
            aux[f"grouped_lm_loss/decade_{decade:02d}"] = torch.stack(values).mean()

        return final_loss, aux, grad_norm
    