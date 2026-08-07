import subprocess
import sys
import argparse
import torch

torch.serialization.add_safe_globals([argparse.Namespace])

CURR_RUN = "06"
BEST_CKPT_NAME = "epoch=143-dice=73.88.ckpt"
# also need to change the filters to match 

training_command = [
    sys.executable,
    "main.py", # not sure if it will be able to find it
    "--exec_mode", "predict",
    "--ckpt_path", f"results/run_{CURR_RUN}/{BEST_CKPT_NAME}",
    "--data", "data/data_for_experiments",
    "--results", f"results/run_{CURR_RUN}/preds",
    "--gpus", "1",
    "--filters", "64", "96", "128", "192", "256", "384",
    "--save_preds",
    # "--brats22_model"
]

subprocess.run(training_command)
