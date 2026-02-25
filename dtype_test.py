import torch
import torch_xla


def main():
    
    x = torch.tensor([1, 2, 3], dtype=torch.float32)
    y = torch.tensor([4, 5, 6], dtype=torch.bfloat16)

    z = x + y
    print(z.dtype)


if __name__ == "__main__":
    main()