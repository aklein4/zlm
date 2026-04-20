import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


FILE = "losses.csv"

RUNS = {
    "fancy-momentum-1b": "iTTT",
    "attn-baseline-theta-1b": "Dense Attn"
}


def main():
    
    df = pd.read_csv(FILE)

    for name, label in RUNS.items():
        data = list(df[f"{name} - loss"].dropna())

        if "attn" in label.lower():
            roll = pd.DataFrame({"k": data}).rolling(100).mean()
            x = (np.arange(len(roll["k"])) + 100) * 32 * (1024 * 32)
        else:
            roll = pd.DataFrame({"k": data}).rolling(25).mean()
            x = (np.arange(len(roll["k"])) + 25) * 128 * (1024 * 32)

        plt.plot(x, roll, label=label)
    
    plt.xlabel("Tokens Seen")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.grid()
    plt.ylim(1.8, 3.0)
    plt.title("Training Loss Over Tokens Seen")
    plt.tight_layout()
    plt.savefig("training_loss.png")
    

if __name__ == "__main__":
    main()
