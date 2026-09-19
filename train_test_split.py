from glob import glob
import os
from subprocess import call
import shutil
# moves 100 samples from training directory to testing directory
# originally, training directory contained all 1,000 samples
# after train test split, will have 900 training and 100 testing samples
# no need for validation set since model automatically creates it during training

print("Current working directory: ", os.getcwd())

# specifies path of training and testing directories 
training_directory =  "data/BraTS-MEN-Train"
testing_directory = "data/BraTS-MEN-Test"

# gets the first 100 training samples and moves to testing 
train_samples = sorted(glob(os.path.join(training_directory, "BraTS*")))[0:100]
print(train_samples)
for sample_index in range(len(train_samples)):
    folder_path = train_samples[sample_index]
    shutil.move(folder_path, testing_directory)

# check training and testing set sizes
print("Training set size: ", len(glob(os.path.join(training_directory, "BraTS*")))) # 900
print("Testing set size: ", len(glob(os.path.join(testing_directory, "BraTS*")))) # 100
