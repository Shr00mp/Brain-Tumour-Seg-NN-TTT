import os
import json
print("Current working directory: ", os.getcwd())

correctly_formated_json = []
with open("results/run_01/logs.jsonl", "r") as logs_file:
    for line in logs_file:
        line = line.strip()
        if line.startswith("DLLL "):
            line = line[5:]
        if line != "":
            correctly_formated_json.append(json.loads(line))

with open("results/run_01/logs_reformatted.json", "w") as logs_file:
    json.dump(correctly_formated_json, logs_file, indent=2)