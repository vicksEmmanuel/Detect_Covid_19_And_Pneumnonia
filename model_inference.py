# Description: This script performs inference on the test dataset and prints the classification metrics
#%%
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torchvision.models as models
from image_dataset import ImageDataset


BATCHSIZE = 4
NUM_CLASSES = 3


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = ImageDataset(path_name="test", transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = models.densenet121(pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
model.load_state_dict(torch.load("model.pth"))
model.to(DEVICE)
model.eval()

true_labels = []
predicted_labels = []

#%%

# Perform inference on the test dataset
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        print(outputs)

        _, predicted_class = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())  # Store true labels
        predicted_labels.extend(predicted_class.cpu().numpy()) 


print(f"True Labels: {true_labels}")
#%%

# Calculate classification metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average="weighted")
recall = recall_score(true_labels, predicted_labels, average="weighted")
f1 = f1_score(true_labels, predicted_labels, average="weighted")
confusion = confusion_matrix(true_labels, predicted_labels)

#%%

# Print classification metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print("Confusion Matrix:")
print(confusion)

#%%