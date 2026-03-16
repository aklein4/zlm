import torch

import torch_xla

from trainers.base_trainer import BaseTrainer
from models.fo_ittt import FoItttModel
from utils.loss_utils import lm_loss_fn
import utils.constants as constants
from utils.logging_utils import master_print
from utils.sharding_utils import maybe_shard_with_gradients


class FoItttTrainer(BaseTrainer):
    
    model: FoItttModel


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
    def first_chunk(self, chunk, second_pass=False):

        if second_pass:
            chunk = chunk.repeat(2, 1)
            chunk = maybe_shard_with_gradients(chunk)

        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):

            logits = self.model(
                chunk,
                logits_to_keep=slice(0, -1)
            )[0]

            if second_pass:
                chunk = chunk[:chunk.shape[0]//2]
                logits = logits[:logits.shape[0]//2]

                chunk = maybe_shard_with_gradients(chunk)
                logits = maybe_shard_with_gradients(logits)

            loss = self.loss(chunk, logits)

        loss.backward()
        self.model.update_state()

        return loss


    @torch_xla.compile(full_graph=True)
    def looped_chunks(self, in_chunk, out_chunk, second_pass=False):

        if second_pass:
            in_chunk = in_chunk.repeat(2, 1)
            out_chunk = out_chunk.repeat(2, 1)

            in_chunk = maybe_shard_with_gradients(in_chunk)
            out_chunk = maybe_shard_with_gradients(out_chunk)

        all_chunk = torch.cat([in_chunk, out_chunk], dim=-1)

        with torch.autocast(
            "xla",
            dtype=torch.bfloat16,
            enabled=self.config.trainer.use_autocast,
        ):

            logits = self.model(
                all_chunk,
                logits_to_keep=slice(in_chunk.shape[-1]-1, -1)
            )[0]

            if second_pass:
                all_chunk = all_chunk[:all_chunk.shape[0]//2]
                logits = logits[:logits.shape[0]//2]

                all_chunk = maybe_shard_with_gradients(all_chunk)
                logits = maybe_shard_with_gradients(logits)

            loss = self.loss(
                all_chunk[:, in_chunk.shape[-1]-1:],
                logits
            )

        loss.backward()
        self.model.update_state()

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

        input_ids: torch.LongTensor = batch["input_ids"]
        chunks = torch.split(
            input_ids, self.config.model.chunk_size,
            dim=-1
        )

        # perform the first pass
        self.model.set_second_pass(False)
        
        self.first_chunk(chunks[0])
        torch_xla.sync()
        master_print("First chunk 00 completed.")

        for i in range(1, len(chunks)):
            in_chunk = chunks[i-1]
            out_chunk = chunks[i]
            
            self.looped_chunks(in_chunk, out_chunk)
            torch_xla.sync()

            master_print(f"First chunk {i:02d} completed.")
        
        # perform the second pass
        self.model.set_second_pass(True)
        self.model.finalize_gradients()
        self.model.zero_grad(set_to_none=False)

        # first chunk
        total_loss = self.first_chunk(chunks[0], second_pass=True)
        aux = {
            "lm_loss/chunk_00": total_loss,
        }
        torch_xla.sync()
        master_print("Second chunk 00 completed.")

        # remaining chunks
        for i in range(1, len(chunks)):
            in_chunk = chunks[i-1]
            out_chunk = chunks[i]
            
            loss = self.looped_chunks(in_chunk, out_chunk, second_pass=True)

            aux[f"lm_loss/chunk_{i:02d}"] = loss
            total_loss = total_loss + loss
            torch_xla.sync()
            master_print(f"Chunk {i:02d} completed.")
        
        aux["relative_grad_error"] = self.model.relative_grad_error()

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
    