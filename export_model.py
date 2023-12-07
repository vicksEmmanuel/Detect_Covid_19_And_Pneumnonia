import torch
import numpy as np
import torchvision.models as models
import torch.onnx as onnx

NUM_CLASSES = 3

# Load your PyTorch model
model = models.densenet121(pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = torch.nn.Linear(num_ftrs, NUM_CLASSES)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Create a NumPy array with the same shape as dummy_input
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Export the model to ONNX with the NumPy array as input
input_names = ["input"]
output_names = ["output"]
onnx.export(model, 
            torch.from_numpy(dummy_input),  # Convert the NumPy array back to a PyTorch tensor
            "model.onnx", 
            export_params=True, 
            opset_version=10,
            do_constant_folding=True,
            input_names=input_names, 
            output_names=output_names,
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print("Model exported successfully")
