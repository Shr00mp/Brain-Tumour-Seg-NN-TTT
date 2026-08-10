import subprocess
import sys

training_command = [
    sys.executable,
    "main.py",
    "--task", "01",
    "--scheduler",
    "--learning_rate", "0.0003",
    "--epochs", "170",
    "--fold", "0",
    "--amp",
    "--gpus", "1",
    "--save_ckpt",
    # "--brats22_model",
    "--deep_supervision",
    "--batch_size", "2",
    "--logname", "logs.jsonl",
    "--results", "results/run_07",
    "--ckpt_store_dir", "results/run_07",
    "--filters", "96", "128", "128", "192", "256", "384",
    "--seed", "42",
    "--resume_training",
    "--ckpt_path", "results/run_07/checkpoints/last.ckpt",
]

subprocess.run(training_command)
