import json
import matplotlib.pyplot as plt
import pandas as pd

df = []

with open("results/run_02/logs.jsonl", "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("DLLL "):
            line = line[5:]
        line_as_json = json.loads(line)
        df.append(
            {
                "epoch": line_as_json.get("step")+1,
                "train_loss": line_as_json.get("data").get("Train Loss"),
                "val_loss": line_as_json.get("data").get("Val Loss"),
                "dice": line_as_json.get("data").get("Dice")
            }
        )

df = pd.DataFrame(df)

loss_fig, loss_ax = plt.subplots()
loss_ax.set_xlabel("Epoch")
loss_ax.set_ylabel("Loss")
(train_loss_line,) = loss_ax.plot(
    df["epoch"], df["train_loss"], label="Train Loss"
)
(val_loss_line,) = loss_ax.plot(
    df["epoch"], df["val_loss"], label="Validation loss"
)

loss_ax.legend(
    [train_loss_line, val_loss_line],
    ["Training loss", "Validation loss"]
)

plt.title("Loss plot (Run 02)")
output_path = "results/run_02/loss_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")



dice_fig, dice_ax = plt.subplots()
dice_ax.set_xlabel("Epoch")
dice_ax.set_ylabel("Dice Score")
(dice_line,) = dice_ax.plot(
    df["epoch"], df["dice"], label="Train Loss"
)

loss_ax.legend(
    [dice_line],
    ["Dice Score"]
)

plt.title("Dice plot (Run 02)")
output_path = "results/run_02/dice_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

