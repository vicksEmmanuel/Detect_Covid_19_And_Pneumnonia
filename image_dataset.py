# %%
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os
from pathlib import Path
import cv2
import torch


class ImageDataset(Dataset):
    def __init__(self, path_name, transform=None):
        super().__init__() #Used to call the constructor of the Dataset class.
        self.all_images = self.collect_images(path_name)
        self.transform = transform

    def collect_images(self, folder_path):
        image_paths = []
        for class_name in os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_name)
            print(f"Class path: {class_path}")  # Debug print

            if os.path.isdir(class_path):#Check if the current path is a directory (ignores files)
                for file in os.listdir(class_path):
                    if file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg"):
                        full_path = os.path.join(class_path, file)
                        print(f"Adding file: {full_path}")  # Additional debug print
                        image_paths.append((full_path, class_name)) 
                        #Append the tuple (full_path, class_name) to the image_paths list
        print(f"Collected {len(image_paths)} images from {folder_path}")  # Debug print
        return image_paths

    def __len__(self):
        return len(self.all_images)
    # This method returns the total length of the dataset, i.e. the number of samples contained in the dataset.

    def __getitem__(self, index):#This method returns a sample at the given index. 
        if index < len(self.all_images):
            print(f"Index: {index}, Dataset Size: {len(self.all_images)}")
            try:
                image_path, label = self.all_images[index]
                # every element in all_images is tuple (full_path, class_name) and they are separately valued to image path and value
                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(f"Unable to load image at {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image) #convert to PIL
                if self.transform:
                    image = self.transform(image)
                label_idx = self.label_to_idx(label)
                label_tensor = torch.tensor(label_idx, dtype=torch.long)
                #converts a numeric index to a PyTorch tensor.(basic data structures in PyTorch, similar to NumPy arrays)
                # #dtype=torch.long specifies that the tensor's datatype is a 64-bit integer, which is typically used to represent labels.
                print(f"Loaded image at {image_path} with label {label}")

                return image, label_tensor
            except Exception as e:
                print(f"Error in __getitem__ at index {index}: {e}")
                return None

    def label_to_idx(self, label):
        # Convert label string to a numerical index
        label_mapping = {'covid': 0, 'normal': 1, 'pneumonia': 2}
        return label_mapping[label]
    
    def idx_to_label(self, idx):
        # Reverse mapping from numerical index to label string
        idx_label_mapping = {0: 'covid', 1: 'normal', 2: 'pneumonia'}
        return idx_label_mapping[idx]
    # Mapping between category labels (strings) and corresponding numeric indexes in the dataset
    # Often used to convert category labels into a form that the model can understand, e.g. mapping text labels to numbers for use in the training process.
# %%
