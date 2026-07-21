import subprocess
import sys

training_preprocess = [
    sys.executable,
    "preprocess.py",
    "--task", "01",
    "--ohe",
    "--exec_mode", "training"
]

testing_preprocess = [
    sys.executable,
    "preprocess.py",
    "--task", "02",
    "--ohe",
    "--exec_mode", "training"
]

subprocess.run(training_preprocess)
subprocess.run(testing_preprocess)
