import subprocess
import sys

training_command = [
    sys.executable,
    "main.py",
    "--task", "01",
    "--depth", "6",
    "--filters", "64", "96", "128", "192", "256", "384", "512",
    "--scheduler",
    "--learning_rate", "0.0003",
    "--epochs", "100",
    "--fold", "0",
    "--amp",
    "--gpus", "1",
    "--save_ckpt",
    # "--brats22_model",
    "--deep_supervision",
    "--batch_size", "4",
    "--logname", "logs.jsonl",
    "--results", "results/run_02"
]

subprocess.run(training_command)
