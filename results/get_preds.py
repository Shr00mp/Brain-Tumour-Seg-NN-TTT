import subprocess
import sys
import argparse
import torch
# Specifies command for getting model predictions 

torch.serialization.add_safe_globals([argparse.Namespace])

training_command = [
    sys.executable,
    "main.py", # not sure if it will be able to find it
    "--exec_mode", "predict",
    "--ckpt_path", "results/run_08/checkpoints/epoch=103-dice=75.30.ckpt",
    "--data", "data/04_3d",
    "--results", "results/task04_tta_step1/preds", # predictions will be stored here 
    "--gpus", "1",
    "--save_preds",
    "--filters", "64", "96", "128", "192", "256", "384", # must specify configuration of model that is running
    "--res_block",
    "--depth", "6"
]

subprocess.run(training_command)

