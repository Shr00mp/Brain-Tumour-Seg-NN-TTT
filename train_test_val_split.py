from glob import glob
import os
from subprocess import call
import shutil

print("Current working directory: ", os.getcwd())

training_directory =  "data/BraTS-MEN-Train"
testing_directory = "data/BraTS-MEN-Test"

# Need to move 100 samples from train to test
train_samples = sorted(glob(os.path.join(training_directory, "BraTS*")))[0:100]
print(train_samples)
for sample_index in range(len(train_samples)):
    folder_path = train_samples[sample_index]
    shutil.move(folder_path, testing_directory)

print("Training set size: ", len(glob(os.path.join(training_directory, "BraTS*")))) # 900
print("Testing set size: ", len(glob(os.path.join(testing_directory, "BraTS*")))) # 100

# For previous method: 

# training_directory_images = "data/BraTS-MEN-Train/images"
# training_directory_labels = "data/BraTS-MEN-Train/labels"
# testing_directory_images = "data/BraTS-MEN-Test/images"
# testing_directory_labels = "data/BraTS-MEN-Test/labels"
# val_directory_images = "data/BraTS-MEN-Val/images"
# val_directory_labels = "data/BraTS-MEN-Val/labels"

# Move the first 100 images of train to test
# train_images = sorted(glob(os.path.join(training_directory_images, "BraTS*")))[0:101]
# train_labels = sorted(glob(os.path.join(training_directory_labels, "BraTS*")))[0:101]
# for sample_index in range(0, 101):
#     img_path = train_images[sample_index]
#     lbl_path = train_labels[sample_index]
#     shutil.move(img_path, testing_directory_images)
#     shutil.move(lbl_path, testing_directory_labels)

# Move the next 100 images of train to validation
# train_images = sorted(glob(os.path.join(training_directory_images, "BraTS*")))[0:101]
# train_labels = sorted(glob(os.path.join(training_directory_labels, "BraTS*")))[0:101]
# for sample_index in range(0, 101):
#     img_path = train_images[sample_index]
#     lbl_path = train_labels[sample_index]
#     shutil.move(img_path, val_directory_images)
#     shutil.move(lbl_path, val_directory_labels)


# print("Train img size: ", len(glob(os.path.join(training_directory_images, "BraTS*")))) # 798
# print("Train lbl size: ", len(glob(os.path.join(training_directory_labels, "BraTS*")))) # 798
# print("Test img size: ",len(glob(os.path.join(testing_directory_images, "BraTS*")))) # 101
# print("Test lbl size: ", len(glob(os.path.join(testing_directory_labels, "BraTS*")))) # 101
# print("Val img size: ", len(glob(os.path.join(val_directory_images, "BraTS*")))) # 101
# print("Val lbl size: ", len(glob(os.path.join(val_directory_labels, "BraTS*")))) # 101