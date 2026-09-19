import json
import matplotlib.pyplot as plt
import pandas as pd
# Get the loss plot and plot of the dice score based on values stored in logs.jsonl 

# All folders are in the format of e.g. run_01
# Each of these folders will have the logs.jsonl file which contains loss and dice values per epoch
CURR_RUN = "01"
df = []

with open(f"results/run_{CURR_RUN}/logs.jsonl", "r") as f:
    for line in f:
        line = line.strip()
        if not line: # checks for empty line
            continue

        if line.startswith("DLLL "): # strips start of line 
            line = line[5:]
        line_as_json = json.loads(line)
        step = line_as_json.get("step")
        if step == []:
            continue
        df.append( # stores jsonl values in pandas dataframe for easier plotting 
            {
                "epoch": line_as_json.get("step")+1, # epochs start with 1, not 0
                "train_loss": line_as_json.get("data").get("Train Loss"),
                "val_loss": line_as_json.get("data").get("Val Loss"),
                "dice": line_as_json.get("data").get("Dice")
            }
        )

df = pd.DataFrame(df)

# Get loss plot and store it as e.g. run_01 > loss_plot.png
loss_fig, loss_ax = plt.subplots()
loss_ax.set_xlabel("Epoch")
loss_ax.set_ylabel("Loss")
loss_ax.set_ylim(0.3, 1.0)
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

plt.title(f"Loss plot (Run {CURR_RUN})")
output_path = f"results/run_{CURR_RUN}/loss_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")


# Get dice plot and store it as e.g. run_01 > dice_plot.png
dice_fig, dice_ax = plt.subplots()
dice_ax.set_xlabel("Epoch")
dice_ax.set_ylabel("Dice Score")
dice_ax.set_ylim(40, 85)
(dice_line,) = dice_ax.plot(
    df["epoch"], df["dice"], label="Train Loss"
)

loss_ax.legend(
    [dice_line],
    ["Dice Score"]
)

plt.title(f"Dice plot (Run {CURR_RUN})")
output_path = f"results/run_{CURR_RUN}/dice_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

