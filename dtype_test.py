import torch
import torch_xla


def main():
    
    x = torch.tensor([1, 2, 3], dtype=torch.float32).to('xla')
    y = torch.tensor([4, 5, 6], dtype=torch.bfloat16).to('xla')

    z = x + y

    torch_xla.sync()

    print(z, z.dtype)


if __name__ == "__main__":
    main()