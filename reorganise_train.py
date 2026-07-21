# %%
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from glob import glob
import os
from pathlib import Path
import json
from subprocess import call
import time
from joblib import Parallel, delayed

print("Current working directory: ", os.getcwd())

# %%
# imgs = [
#     nibabel.load(f"data/BraTS-MEN-Train/BraTS-MEN-00008-000/BraTS-MEN-00008-000-{m}.nii.gz").get_fdata().astype(np.float32)[:, :, 23] 
#     for m in ["t1c", "t1n", "t2f", "t2w"]]
# lbl = nibabel.load("data/BraTS-MEN-Train/BraTS-MEN-00008-000/BraTS-MEN-00008-000-seg.nii.gz").get_fdata().astype(np.float32)[:, :, 23]
# print(np.any(lbl != 0))

# fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 15))
# for i, img in enumerate(imgs):
#     ax[i].imshow(img, cmap='gray')
#     ax[i].axis('off')
# ax[-1].imshow(lbl, vmin=0, vmax=4)
# ax[-1].axis('off')
# plt.tight_layout()            
# plt.show()

# %%
# entire = nibabel.load("data/BraTS-MEN-Train/BraTS-MEN-00008-000/BraTS-MEN-00008-000-seg.nii.gz").get_fdata()
# np.where(entire == 3)

# %%
training_directory = "data/BraTS-MEN-Train"
testing_directory = "data/BraTS-MEN-Test"

# %%
# t1c = t1c
# t1n = t1
# t2f = flair
# t2w = t2

def load_nifty(directory, example_id, suffix):
    return nibabel.load(os.path.join(directory, example_id + "-" + suffix + ".nii.gz"))


def load_channels(d, example_id):
    return [load_nifty(d, example_id, suffix) for suffix in ["t1c", "t1n", "t2f", "t2w"]]


def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)


def prepare_nifty(d):
    example_id = d.split("/")[-1]
    t2f, t1n, t1c, t2w = load_channels(d, example_id)
    affine, header = t2f.affine, t2f.header
    vol = np.stack([get_data(t2f), get_data(t1n), get_data(t1c), get_data(t2w)], axis=-1)
    vol = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
    nibabel.save(vol, os.path.join(d, example_id + ".nii.gz"))

    if os.path.exists(os.path.join(d, example_id + "-seg.nii.gz")):
        seg = load_nifty(d, example_id, "seg")
        affine, header = seg.affine, seg.header
        vol = get_data(seg, "unit8")
        vol[vol == 4] = 3
        seg = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
        nibabel.save(seg, os.path.join(d, example_id + "-seg.nii.gz"))


def prepare_dirs(data, train):
    img_path, lbl_path = os.path.join(data, "images"), os.path.join(data, "labels")
    call(f"mkdir {img_path}", shell=True)
    if train:
        call(f"mkdir {lbl_path}", shell=True)
    dirs = glob(os.path.join(data, "BraTS*"))
    for d in dirs:
        if "-" in d.split("/")[-1]:
            files = glob(os.path.join(d, "*.nii.gz"))
            for f in files:
                if "t1c" in f or "t1n" in f or "t2f" in f or "t2w" in f:
                    continue
                if "-seg" in f:
                    call(f"mv {f} {lbl_path}", shell=True)
                else:
                    call(f"mv {f} {img_path}", shell=True)
        call(f"rm -rf {d}", shell=True)


def prepare_dataset_json(data, train):
    images, labels = glob(os.path.join(data, "images", "*")), glob(os.path.join(data, "labels", "*"))
    images = sorted([img.replace(data + "/", "") for img in images])
    labels = sorted([lbl.replace(data + "/", "") for lbl in labels])

    modality = {"0": "t2f", "1": "t1n", "2": "t1c", "3": "t2w"}
    labels_dict = {"0": "background", "1": "NETC", "2": "SNFH", "3": "ET"}
    if train:
        key = "training"
        data_pairs = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs = [{"image": img} for img in images]

    dataset = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs,
    }

    with open(os.path.join(data, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)

def run_parallel(func, args):
    return Parallel(n_jobs=os.cpu_count())(delayed(func)(arg) for arg in args)


def prepare_dataset(data, train):
    print(f"Preparing BraTS21 dataset from: {data}")
    start = time.time()
    run_parallel(prepare_nifty, sorted(glob(os.path.join(data, "BraTS*"))))
    prepare_dirs(data, train)
    prepare_dataset_json(data, train)
    end = time.time()
    print(f"Preparing time: {(end - start):.2f}")

prepare_dataset(training_directory, True)
print("Finished!")

# %%
# glob(os.path.join(training_directory, "BraTS*"))


