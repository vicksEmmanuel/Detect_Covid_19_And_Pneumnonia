# %%
from datetime  import datetime
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import torchvision.models as models
from image_dataset import ImageDataset
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from image_dataset import ImageDataset

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #Specifies whether to use GPU ('cuda') or CPU ('cpu') for training; "cuda" is prior to "cpu"
EPOCHS = 15 #Number of training epochs.
BATCHSIZE = 4 #Batch size used during training.
NUM_CLASSES = 3 #Number of output classes
#Constants and Configuration

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
#create the training set object, use the path name to return the image and use the created transform to process the data
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
    # taking as input two datasets, calculates the number of samples in each of them for each category
    # ploting a bar chart showing the distribution of the categories via the seaborn library
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
    #taking a dataset as input, and the number of images to display for each category (default is 3)
    #It iterates through the dataset, selects the top few images for each category, and displays those images.
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
#Loading the pre-trained DenseNet-121 model(121 layers of parameters were used)
num_ftrs = model.classifier.in_features
#retrieving the number of input features (or neurons) in the last fully connected classifier of the pre-trained DenseNet-121 model
model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
#replacing the existing classifier; a linear layer with input features (num_ftrs) and output features equal to the number of classes in your specific problem (NUM_CLASSES).
model = model.to(DEVICE)#move to device(CPU or GPU)


criterion = nn.CrossEntropyLoss()#instantiated
#the loss function which is commonly used to measure the difference between the model's output and the actual labels. 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#creating a Stochastic Gradient Descent (SGD) optimizer which is to minimize the loss by adjusting the model's weights. 
#lr=0.01 sets the learning rate, and momentum=0.9 is a hyperparameter for the SGD optimizer, helping to accelerate gradient updates.
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#Learning rate schedulers dynamically adjust the learning rate during training to facilitate better convergence to the optimal solution.
#StepLR scheduler is used, which multiplies the learning rate by gamma every step_size epochs. 

train_losses, val_losses = [], []
#store training and validation losses during the training process.
# %%

start_time = datetime.now()
print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")



# %% 
for epoch in range(EPOCHS): #training loop:train EPOCHS times.Each epoch corresponds to a complete pass through the entire training dataset.
    model.train()
    #Sets the model to training mode.
    running_train_loss, running_val_loss = 0.0, 0.0

    for i, data in enumerate(train_data_loader):
        image, label = data
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        # Clears the gradients of all optimized parameters. This is necessary before computing gradients in the backward pass.

        output = model(image.float())
        # Forward pass: computes the output predictions of the model for the given input.

        train_loss = criterion(output.float(), label.long())
        #Computes the training loss by comparing the model's output with the actual labels

        train_loss.backward()
        optimizer.step()
        #Backward pass: computes the gradients of the model parameters with respect to the loss and updates the model's weights using the optimizer.

        running_train_loss += train_loss.item()
        print(f"Epoch: {epoch}, Iteration: {i}, Loss: {train_loss.item()}")
        #Updates the running training loss and prints the current loss for monitoring.

    exp_lr_scheduler.step()  # Update the learning rate

    train_losses.append(running_train_loss)

    model.eval()
    #Sets the model to evaluation mode. This is important because certain layers may behave differently during evaluation.

    #Similar to the training loop, but this loop evaluates the model on the validation dataset without performing backpropagation.
    with torch.no_grad():#used to disable gradient computation, saving memory and speeding up the inference process
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
#Plot training and validation loss curves

end_time = datetime.now()
print(f"Training completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


total_duration = end_time - start_time
print(f"Total training duration: {total_duration}")

# %%
torch.save(model.state_dict(), "model.pth")


# %%
