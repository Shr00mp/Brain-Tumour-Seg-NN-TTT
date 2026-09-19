import subprocess
import sys
# specifies commands for training the UNet 

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
    # "--brats22_model", # causes brats22_model to be used. otherwise DynUNet is used 
    "--deep_supervision",
    "--batch_size", "2",
    "--logname", "logs.jsonl",
    "--results", "results/run_17",
    "--ckpt_store_dir", "results/run_17",
    "--depth", "6",
    "--filters", "64", "96", "128", "192", "256", "384",
    "--res_block",
    "--seed", "42"
]

subprocess.run(training_command)
