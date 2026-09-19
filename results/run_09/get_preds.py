import subprocess
import sys
import argparse
import torch

torch.serialization.add_safe_globals([argparse.Namespace])

training_command = [
    sys.executable,
    "main.py", # not sure if it will be able to find it
    "--exec_mode", "predict",
    "--ckpt_path", "results/run_09/checkpoints/epoch=139-dice=75.77.ckpt",
    "--data", "data/data_for_experiments",
    "--results", "results/run_09/preds",
    "--gpus", "1",
    "--save_preds",
    "--filters", "32", "64", "96", "128", "192", "256", "384",
    "--res_block",
    "--depth", "7",
]

subprocess.run(training_command)
