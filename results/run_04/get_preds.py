import subprocess
import sys
import argparse
import torch

torch.serialization.add_safe_globals([argparse.Namespace])

training_command = [
    sys.executable,
    "main.py", # not sure if it will be able to find it
    "--exec_mode", "predict",
    "--ckpt_path", "results/run_04/checkpoints/last.ckpt",
    "--data", "data/data_for_experiments",
    "--results", "results/run_04/preds",
    "--gpus", "1",
    "--save_preds"
]

subprocess.run(training_command)
