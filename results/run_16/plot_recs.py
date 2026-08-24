import numpy as np
import nibabel as nib
import glob as glob
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

sample_slice_dict = {
    "00155": 95,
    "00156": 100,
    "00157": 45,
    "00159": 55,
    "00161": 50,
    "00163": 47,
    "00164": 130,
    "00165": 100,
    "00166": 100,
    "00167": 100,
    "00168": 55,
    "00169": 60,
    "00170": 82,
    "00172": 40,
    "00173": 120,
    "00174": 57,
    "00175": 117,
    "00176": 105,
    "00177": 110,
    "00178": 125
}

rec_paths = glob.glob(os.path.join("results/run_16/preds/predictions_epoch=148-dice=73_01_task=01_fold=0/rec", "*.npy"))
scan_paths = glob.glob(os.path.join("data/BraTS-MEN-Train/images", "BraTS*"))   

rec_path = ""
scan_path = ""

for sample_num, slice_num in sample_slice_dict.items():
    for j in rec_paths:
        if f"{sample_num}" in j:
            rec_path = j
    for s in scan_paths:
        if f"{sample_num}" in s:
            scan_path = s

    # load reconstruction
    rec_data = np.load(rec_path)
    rec_slice = rec_data.transpose(2, 3, 1, 0)
    print("Rec_slice shape: ", rec_slice.shape)
    # rec shape (5, 155, 240, 240)
    # I think the last dimension is the one hot encoding for the foreground 
    rec_t1_slice = rec_slice[:, :, slice_num, 0]
    rec_t1c_slice = rec_slice[:, :, slice_num, 1]
    rec_t2_slice = rec_slice[:, :, slice_num, 2]
    rec_flair_slice = rec_slice[:, :, slice_num, 3]
    rec_t1_slice = np.rot90(rec_t1_slice, k=3)[:, ::-1]
    rec_t1c_slice = np.rot90(rec_t1c_slice, k=3)[:, ::-1]
    rec_t2_slice = np.rot90(rec_t2_slice, k=3)[:, ::-1]
    rec_flair_slice = np.rot90(rec_flair_slice, k=3)[:, ::-1]
    print("Shape of rec slice: ", rec_flair_slice.shape)

    # load original scan 
    scan = nib.load(scan_path)
    scan_data = scan.get_fdata()
    scan_slice = scan_data.transpose(1, 0, 2, 3)

    orig_t1_slice = scan_slice[:, :, slice_num, 0]
    orig_t1c_slice = scan_slice[:, :, slice_num, 1]
    orig_t2_slice = scan_slice[:, :, slice_num, 2]
    orig_flair_slice = scan_slice[:, :, slice_num, 3]
    orig_t1_slice = np.rot90(orig_t1_slice, k=3)[:, ::-1]
    orig_t1c_slice = np.rot90(orig_t1c_slice, k=3)[:, ::-1]
    orig_t2_slice = np.rot90(orig_t2_slice, k=3)[:, ::-1]
    orig_flair_slice = np.rot90(orig_flair_slice, k=3)[:, ::-1]
    print("Orig scan shape: ", orig_flair_slice.shape)

    plt.figure(figsize=(20, 5))
    plt.suptitle(f"Slice number: {slice_num}")

    plt.subplot(1, 8, 1)
    plt.imshow(orig_t1_slice, cmap="gray", origin='lower')
    plt.axis("off")
    plt.title("Original T1", fontsize=15)

    plt.subplot(1, 8, 2)
    plt.imshow(rec_t1_slice, cmap="gray", origin='lower')
    plt.axis("off")
    plt.title("Reconstructed T1", fontsize=15)

    plt.subplot(1, 8, 3)
    plt.imshow(orig_t1c_slice, cmap="gray", origin='lower')
    plt.axis("off")
    plt.title("Original T1c", fontsize=15)

    plt.subplot(1, 8, 4)
    plt.imshow(rec_t1c_slice, cmap="gray", origin='lower')
    plt.axis("off")
    plt.title("Reconstructed T1c", fontsize=15)

    plt.subplot(1, 8, 5)
    plt.imshow(orig_t2_slice, cmap="gray", origin='lower')
    plt.axis("off")
    plt.title("Original T2", fontsize=15)

    plt.subplot(1, 8, 6)
    plt.imshow(rec_t2_slice, cmap="gray", origin='lower')
    plt.axis("off")
    plt.title("Reconstructed T2", fontsize=15)

    plt.subplot(1, 8, 7)
    plt.imshow(orig_flair_slice, cmap="gray", origin='lower')
    plt.axis("off")
    plt.title("Original FLAIR", fontsize=15)

    plt.subplot(1, 8, 8)
    plt.imshow(rec_flair_slice, cmap="gray", origin='lower')
    plt.axis("off")
    plt.title("Reconstructed FLAIR", fontsize=15)

    plt.tight_layout()

    plt.savefig(f"results/run_16/preds/rec_plots/{sample_num}.png")
