import subprocess
import sys
import argparse
import torch

torch.serialization.add_safe_globals([argparse.Namespace])

training_command = [
    sys.executable,
    "main.py", # not sure if it will be able to find it
    "--exec_mode", "predict",
    "--ckpt_path", "results/run_06/checkpoints/epoch=143-dice=73.88.ckpt",
    "--data", "data/data_for_experiments",
    "--results", "results/run_06/preds",
    "--gpus", "1",
    "--save_preds",
    "--filters", "96", "96", "128", "192", "256", "320",
    "--res_block"
]

subprocess.run(training_command)
