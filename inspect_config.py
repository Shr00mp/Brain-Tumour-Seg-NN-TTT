import pickle

with open("data/01_3d/config.pkl", "rb") as f:
    configs = pickle.load(f)

print(configs)
# {'patch_size': [128, 128, 128], 'spacings': [1.0, 1.0, 1.0], 'n_class': 4, 'in_channels': 5}