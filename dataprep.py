import os
import random
from shutil import copy2

def create_folders():
    FOLDERS = ["train", "test", "val"]
    SUBFOLDERS = ["covid", "pneumonia", "normal"]

    for folder in FOLDERS:
        for subfolder in SUBFOLDERS:
            path = os.path.join(folder, subfolder) #concatenate path
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Created folder: {path}")
            else:
                print(f"Folder {path} already exists.")
            #check if the path is existed if not, execute ecursively creates the directory and all its non-existing parents.


def distribute_files(src_folder, class_name, file_list, train_pct=0.6, val_pct=0.2):
    total_files = len(file_list)
    train_count = int(total_files * train_pct)
    val_count = int(total_files * val_pct)

    train_files = file_list[:train_count]
    val_files = file_list[train_count:train_count + val_count]
    test_files = file_list[train_count + val_count:]

    for file in train_files:
        copy2(os.path.join(src_folder, file), os.path.join('train', class_name, file))

    for file in val_files:
        copy2(os.path.join(src_folder, file), os.path.join('val', class_name, file))

    for file in test_files:
        copy2(os.path.join(src_folder, file), os.path.join('test', class_name, file))

    #For the line 28-35, the code is used to copy the splited file to the specfic path


def process_images():
    create_folders()

    image_dataset_path = "image_dataset"
    classes = ["covid", "normal", "pneumonia"]

    for class_name in classes:
        folder_path = os.path.join(image_dataset_path, class_name)
        if os.path.isdir(folder_path):
            file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            # the f is added if it is a file
            random.shuffle(file_list)

            # Distribute files to train, val, and test folders
            distribute_files(folder_path, class_name, file_list)

process_images()
