# %%
from datetime  import datetime
import cv2
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import torchvision.models as models
from image_dataset import ImageDataset
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import urllib.request
import ssl
import seaborn as sns
import matplotlib.pyplot as plt

from image_dataset import ImageDataset

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCHSIZE = 4
NUM_CLASSES = 3

# %%

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transform for validation (without RandomHorizontalFlip)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = ImageDataset(path_name="train",transform=train_transform)
train_data_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)

val_dataset = ImageDataset(path_name="val", transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=True)


# %%

def count_samples(dataset):
    # Count the number of samples in each class
    class_counts = {"covid": 0, "normal": 0, "pneumonia": 0}
    
    for i in range(len(dataset)):
        _, label_idx = dataset[i]
        label = dataset.idx_to_label(label_idx.item())  # Convert tensor to string label
        class_counts[label] += 1
    
    return class_counts



def plot_class_distribution(dataset_1, dataset_2):
    train_counts = count_samples(dataset_1)
    val_counts = count_samples(dataset_2)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    sns.barplot(x=list(train_counts.keys()), y=list(train_counts.values()), ax=ax[0])
    ax[0].set_title("Training Set")
    ax[0].set_ylabel("Number of Samples")
    ax[0].set_xlabel("Class")

    sns.barplot(x=list(val_counts.keys()), y=list(val_counts.values()), ax=ax[1])
    ax[1].set_title("Validation Set")
    ax[1].set_ylabel("Number of Samples")
    ax[1].set_xlabel("Class")

    plt.tight_layout()
    plt.show()


def show_images_from_classes(dataset, num_images=3):
    # Store images for each class
    images_to_show = {'covid': [], 'normal': [], 'pneumonia': []}

    for image, label_idx in dataset:
        label = dataset.idx_to_label(label_idx.item())  # Convert tensor to string label
        if len(images_to_show[label]) < num_images:
            image_np = image.numpy()
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
            images_to_show[label].append(image_np)

        if all(len(images) == num_images for images in images_to_show.values()):
            break

    # Plotting
    fig, axes = plt.subplots(num_images, len(images_to_show), figsize=(15, 5))

    for i, (label, images) in enumerate(images_to_show.items()):
        for j, image in enumerate(images):
            ax = axes[j][i] if num_images > 1 else axes[i]
            ax.imshow(np.transpose(image, (1, 2, 0)))  # Convert image back to HxWxC format
            ax.set_title(f"Class: {label}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()


plot_class_distribution(train_dataset, val_dataset)

show_images_from_classes(train_dataset)



# %%

model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_losses, val_losses = [], []

# %%

start_time = datetime.now()
print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")



# %%
for epoch in range(EPOCHS):
    model.train()
    running_train_loss, running_val_loss = 0.0, 0.0

    for i, data in enumerate(train_data_loader):
        image, label = data
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()

        output = model(image.float())

        train_loss = criterion(output.float(), label.long())

        train_loss.backward()
        optimizer.step()

        running_train_loss += train_loss.item()
        print(f"Epoch: {epoch}, Iteration: {i}, Loss: {train_loss.item()}")

    exp_lr_scheduler.step()  # Update the learning rate

    train_losses.append(running_train_loss)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            image, label = data
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            output = model(image.float())

            val_loss = criterion(output.float(), label.long())
            running_val_loss += val_loss.item()

    val_losses.append(running_val_loss)




# %%
sns.lineplot(x=range(len(train_losses)), y=train_losses).set_title("Training Loss")
sns.lineplot(x=range(len(val_losses)), y=val_losses).set_title("Training Loss")


end_time = datetime.now()
print(f"Training completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


total_duration = end_time - start_time
print(f"Total training duration: {total_duration}")

# %%
torch.save(model.state_dict(), "model.pth")


# %%
