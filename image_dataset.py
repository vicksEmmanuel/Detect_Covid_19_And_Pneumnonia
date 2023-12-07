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
        super().__init__()
        self.all_images = self.collect_images(path_name)
        self.transform = transform

    def collect_images(self, folder_path):
        image_paths = []
        for class_name in os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_name)
            print(f"Class path: {class_path}")  # Debug print

            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg"):
                        full_path = os.path.join(class_path, file)
                        print(f"Adding file: {full_path}")  # Additional debug print
                        image_paths.append((full_path, class_name))
        print(f"Collected {len(image_paths)} images from {folder_path}")  # Debug print
        return image_paths

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        if index < len(self.all_images):
            print(f"Index: {index}, Dataset Size: {len(self.all_images)}")
            try:
                image_path, label = self.all_images[index]
                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(f"Unable to load image at {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                if self.transform:
                    image = self.transform(image)
                label_idx = self.label_to_idx(label)
                label_tensor = torch.tensor(label_idx, dtype=torch.long)

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


# %%
