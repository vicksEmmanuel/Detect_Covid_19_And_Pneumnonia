import pickle
import torch
from pathlib import Path
import cv2
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# Define the path to your random image
image_path = "train/covid/1B734A89-A1BF-49A8-A1D3-66FAFA4FAC5D-768x635.jpeg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)

image = test_transform(image)  # Apply preprocessing


# Move the preprocessed image to the appropriate device
image = image.to(DEVICE)

model = models.densenet121(pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
model.load_state_dict(torch.load("model.pth"))
model.to(DEVICE)
model.eval()

# Make predictions
with torch.no_grad():
    output = model(image.unsqueeze(0))  # Unsqueeze to add a batch dimension (batch size of 1)
    _, predicted_class = torch.max(output, 1)

# Convert the predicted class index to a class label (e.g., "covid", "normal", "pneumonia")
class_labels = ["covid", "normal", "pneumonia"]
predicted_label = class_labels[predicted_class.item()]

# Print the predicted label
print(f"Predicted Class: {predicted_label}")


# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)

traced_model = torch.jit.trace(model, image.unsqueeze(0))

# Save the traced model
torch.jit.save(traced_model, "mobile.pt")