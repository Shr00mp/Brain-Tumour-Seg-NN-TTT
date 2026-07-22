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
    "--epochs", "30",
    "--fold", "0",
    "--amp",
    "--gpus", "1",
    "--save_ckpt"
]

subprocess.run(training_command)
