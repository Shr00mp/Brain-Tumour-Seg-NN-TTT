import numpy as np
import nibabel as nib
import glob as glob
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

label_colors = {
    0: "black",
    1: "red",
    2: "blue",
    3: "green"
}

label_names = {
    0: "Background",
    1: "NETC",
    2: "SNFH",
    3: "ET"
}

cmap = ListedColormap([label_colors[i] for i in range(4)])
handles = [mpatches.Patch(color=label_colors[i], label=label_names[i]) for i in range(4)]

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

prediction_paths = glob.glob(os.path.join("results/run_12/preds/predictions_epoch=107-dice=72_55_task=01_fold=0/seg", "*.npy"))
orig_paths = glob.glob(os.path.join("data/BraTS-MEN-Train/labels", "BraTS*"))
scan_paths = glob.glob(os.path.join("data/BraTS-MEN-Train/images", "BraTS*"))
# orig_paths = glob.glob(os.path.join("data/data_for_experiments", "*_orig_lbl.npy"))

pred_path = ""
orig_path = ""
scan_path = ""
for sample_num, slice_num in sample_slice_dict.items():
    for p in prediction_paths: 
        if f"{sample_num}" in p:
            pred_path = p
    for o in orig_paths:
        if f"{sample_num}" in o:
            orig_path = o
    for s in scan_paths:
        if f"{sample_num}" in s:
            scan_path = s

    prob_pred = np.load(pred_path)
    seg_mask = np.argmax(prob_pred, axis=0)
    # orig_data = np.load(orig_path)
    seg_mask = seg_mask.transpose(2, 1, 0)
    orig = nib.load(orig_path) 
    orig_data = orig.get_fdata()

    scan = nib.load(scan_path)
    scan_data = scan.get_fdata()
    scan_slice = scan_data.transpose(1, 0, 2, 3)

    tumour_slices = np.where(orig_data.any(axis=(0, 1)))[0]
    if slice_num == 0: slice_num = tumour_slices[8]

    t1_slice = scan_slice[:, :, slice_num, 0]
    t1c_slice = scan_slice[:, :, slice_num, 1]
    t2_slice = scan_slice[:, :, slice_num, 2]
    flair_slice = scan_slice[:, :, slice_num, 3]
    t1_slice = np.rot90(t1_slice, k=3)[:, ::-1]
    t1c_slice = np.rot90(t1c_slice, k=3)[:, ::-1]
    t2_slice = np.rot90(t2_slice, k=3)[:, ::-1]
    flair_slice = np.rot90(flair_slice, k=3)[:, ::-1]
    
    print("Seg mask shape: ", seg_mask.shape)
    print("Orig shape: ", orig_data.shape)
    print("Data sample shape: ", scan_data.shape)
    print(f"Slices with tumour for {sample_num}: ", tumour_slices)

    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Slice number: {slice_num}")

    plt.subplot(1, 6, 1)
    plt.imshow(t1_slice, cmap="gray", origin='lower')
    plt.axis("off")
    plt.title("T1 sample", fontsize=15)

    plt.subplot(1, 6, 2)
    plt.imshow(t1c_slice, cmap="gray", origin='lower')
    plt.axis("off")
    plt.title("T1c sample", fontsize=15)

    plt.subplot(1, 6, 3)
    plt.imshow(t2_slice, cmap="gray", origin='lower')
    plt.axis("off")
    plt.title("T2 sample", fontsize=15)

    plt.subplot(1, 6, 4)
    plt.imshow(flair_slice, cmap="gray", origin='lower')
    plt.axis("off")
    plt.title("FLAIR sample", fontsize=15)

    plt.subplot(1, 6, 5)
    plt.imshow(orig_data[:, :, slice_num], cmap=cmap, origin='lower')
    plt.axis("off")
    plt.title("ground truth", fontsize=15)

    plt.subplot(1, 6, 6)
    plt.imshow(seg_mask[:, :, slice_num], cmap=cmap, origin='lower')
    plt.axis("off")
    plt.title("predicted label", fontsize=15)

    plt.legend(handles=handles, bbox_to_anchor=(1.0, 0.0), loc="best")

    plt.tight_layout()

    plt.savefig(f"results/run_12/preds/seg_plots/{sample_num}.png")
