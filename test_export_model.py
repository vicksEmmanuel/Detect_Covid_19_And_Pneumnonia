import cv2
import numpy as np
import onnxruntime as ort

import torch
from pathlib import Path
import cv2
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image

import onnx
import onnxruntime as ort

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

def load_image(image_path):
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

    image = test_transform(image)
    return image

image_path = 'val-1/covid/COVID-19 (271).jpg'
image = load_image(image_path)

# Perform inference
class_mapping = {0: 'covid', 1: 'normal', 2: 'pneumonia'}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

ort_session = ort.InferenceSession("model.onnx")

# Prepare the input NumPy array with batch dimension
input_data = np.array(image)[np.newaxis, :]  # Add a batch dimension

# Get the input name from the ONNX model
input_name = ort_session.get_inputs()[0].name

print(f"Input name: {input_name}, Input shape: {input_data.shape}")

# Create a dictionary with the input name as the key and the input data as the value
input_dict = {input_name: input_data}

# Perform inference
outputs = ort_session.run(None, input_dict)

# Extract the logits
logits = outputs[0][0]  # Assuming a single batch

# Apply softmax to convert logits to probabilities
probabilities = softmax(logits)

# Find the class index with the highest probability
class_index = np.argmax(probabilities)

# Map index to class name
predicted_class = class_mapping[class_index]

print(f"Predicted Class: {predicted_class}, Probabilities: {probabilities}")