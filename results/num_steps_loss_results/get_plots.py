import json
import matplotlib.pyplot as plt
import pandas as pd


with open(f"results/num_steps_loss_results/loss_logs.jsonl", "r") as f:
    all_losses = json.load(f)

for sample in all_losses.keys():
    losses = all_losses[sample]
    plt.figure()
    plt.plot(
        range(1, len(losses)),
        losses[1:]
    )
    plt.xlabel("Gradient steps")
    plt.ylabel("Reconstruction loss")
    plt.title(f"Reconstruction loss plot for {sample}")
    plt.xticks(range(1, len(losses)))

    plt.savefig(f"results/num_steps_loss_results/plots/{sample}", dpi=300, bbox_inches="tight")

