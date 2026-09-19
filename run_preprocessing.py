import subprocess
import sys
# specifies commands for running preprocessing on specific tasks
# the folder that corresponds to each task is specified in data_preprocessing > configs.py

training_preprocess = [
    sys.executable,
    "preprocess.py",
    "--task", "03",
    "--ohe",
    "--exec_mode", "training"
]

testing_preprocess = [
    sys.executable,
    "preprocess.py",
    "--task", "04",
    "--ohe",
    "--exec_mode", "training"
]

subprocess.run(training_preprocess)
subprocess.run(testing_preprocess)
