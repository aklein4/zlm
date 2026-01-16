import torch


class IMLCollator:

    def __init__(
        self,
        iml_test_fraction,
    ):
        self.iml_test_fraction = iml_test_fraction
        
    
    def __call__(
        self,
        batch,
    ):  
        
        all_ids = []
        for x in batch:
            all_ids.append(
                torch.tensor(x["input_ids"]).long().flatten()
            )
        all_ids = torch.stack(all_ids, dim=0)

        input_ids, output_ids = torch.chunk(
            all_ids, 2, dim=-1,
        )

        coin = torch.rand(
            input_ids.shape[0], 1, device=input_ids.device, dtype=torch.float32
        ) < 0.5
        input_ids, output_ids = (
            torch.where(
                coin,
                input_ids,
                output_ids,
            ),
            torch.where(
                coin,
                output_ids,
                input_ids,
            ),
        )

        test_bs = input_ids.shape[0] // self.iml_test_fraction
        other_ids = input_ids[:test_bs]
        input_ids = input_ids[test_bs:]
        output_ids = output_ids[test_bs:]

        return {
            "input_ids": input_ids,
            "output_ids": output_ids,
            "other_ids": other_ids,
        }
