import torch


N = 8
T = 6


def main():
    
    X = torch.randn(T, N)
    B = torch.rand(N).requires_grad_(True)

    print("")
    for i in range(1, len(X)+1):

        running = 0
        for x in X[:i]:
            running = B * running + x

        loss = running.sum()
        loss.backward()
    
        print(f"Gradient {i}:", B.grad)
        B.grad.zero_()

    print("")
    for i in range(1, len(X)+1):

        conv = 0
        running = 0
        for x in X[:i]:

            conv = B * conv + running
            running = B * running + x
            
        # should be the same as the gradient
        print(f"Conv {i}:", conv)
    

if __name__ == "__main__":
    main()
