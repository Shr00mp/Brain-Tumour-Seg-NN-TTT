import subprocess
import sys
import argparse
import torch

torch.serialization.add_safe_globals([argparse.Namespace])

training_command = [
    sys.executable,
    "main.py", # not sure if it will be able to find it
    "--exec_mode", "predict",
    "--ckpt_path", "results/run_16/checkpoints/epoch=148-dice=73.01.ckpt",
    "--data", "data/data_for_experiments",
    "--results", "results/run_16/preds",
    "--gpus", "1",
    "--save_preds",
    "--filters", "64", "96", "128", "192", "256", "384",
    "--res_block",
    "--depth", "6"
]

subprocess.run(training_command)
