from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn
import io

def create_app():
    app = Flask(__name__)

    # Load your model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 3
    model = models.densenet121(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load("model.pth"))
    model.to(DEVICE)
    model.eval()

    @app.route('/')
    def welcome():
        return "Welcome to the API!"

    @app.route('/predict', methods=['POST'])
    def predict():
        if request.method == 'POST':
            # Convert the Image from request to a PIL Image
            image = Image.open(io.BytesIO(request.files['image'].read()))

            # Preprocess the image and prepare it for your model
            preprocess =  transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)

            # Run inference
            with torch.no_grad():
                output = model(input_batch)  # Unsqueeze to add a batch dimension (batch size of 1)
                _, predicted_class = torch.max(output, 1)

            # Return the response as JSON

            class_labels = ["covid", "normal", "pneumonia"]
            predicted_label = class_labels[predicted_class.item()]

            return jsonify(result=predicted_label)

    return app



# if __name__ == '__main__':
#     app = create_app()
#     app.run(debug=True)
