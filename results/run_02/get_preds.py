import subprocess
import sys
import argparse
import torch

torch.serialization.add_safe_globals([argparse.Namespace])

training_command = [
    sys.executable,
    "main.py", # not sure if it will be able to find it
    "--exec_mode", "predict",
    "--ckpt_path", "results/run_02/checkpoints/epoch=97-dice=76.02.ckpt",
    "--data", "data/data_for_experiments",
    "--results", "results/run_02/preds",
    "--gpus", "1",
    "--save_preds",
    "--filters", "64", "96", "128", "192", "256", "384",
]

subprocess.run(training_command)
