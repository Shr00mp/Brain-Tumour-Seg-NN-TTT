import subprocess
import sys

training_command = [
    sys.executable,
    "main.py",
    "--task", "01",
    "--scheduler",
    "--learning_rate", "0.0003",
    "--epochs", "150",
    "--fold", "0",
    "--amp",
    "--gpus", "1",
    "--save_ckpt",
    # "--brats22_model",
    "--deep_supervision",
    "--batch_size", "4",
    "--logname", "logs.jsonl",
    "--results", "results/run_06",
    "--ckpt_store_dir", "results/run_06",
    "--filters", "96", "96", "128", "192", "256", "384", "512"
]

subprocess.run(training_command)
