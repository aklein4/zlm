import torch


class SingleSequenceCollator:

    def __init__(
        self,
        sequence_length: int,
        pad_token_id: int,
        vocab_size: int,
    ):
        """
        A collator for pre-tokenized data without equal length, with truncation and right-padding to a fixed length.
        """
        
        self.sequence_length = sequence_length

        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

    
    def __call__(
        self,
        batch,
    ):

        input_ids = []
        for x in batch:

            in_ids = torch.tensor(x["input_ids"]).long().flatten()
            input_ids.append(in_ids)

        # pad to ragged length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,   
        )
        input_ids = input_ids[:, :self.sequence_length]
        
        # pad to sequence length
        pad = torch.full(
            (input_ids.shape[0], self.sequence_length - input_ids.shape[1]),
            self.pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        input_ids = torch.cat(
            [
                input_ids,
                pad
            ],
            dim=1
        )

        invalid = (
            (input_ids < 0)
            | ((input_ids >= self.vocab_size) & (input_ids != self.pad_token_id))
        )
        if invalid.any():
            raise ValueError("Batch contains token IDs outside the configured vocabulary.")

        attention_mask = input_ids != self.pad_token_id
        labels = torch.where(
            attention_mask,
            input_ids,
            torch.full_like(input_ids, -100),
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
