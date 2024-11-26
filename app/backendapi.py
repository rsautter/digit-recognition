from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import io
import numpy as np

# Load the saved model
class LargerCNN(nn.Module):
    def __init__(self):
        super(LargerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LargerCNN()
model.load_state_dict(torch.load("../models/mnist_larger_cnn.pth", map_location=device))
model.to(device)
model.eval()

# Define transformation for input
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Digit Recognition API"}

@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image = Image.open(io.BytesIO(await file.read())).convert("L")  # Convert to grayscale

        # Check if the image is blank
        if np.array(image).sum() == 255 * image.size[0] * image.size[1]:
            return JSONResponse(content={"error": "Image is blank. Please provide a valid digit."}, status_code=400)

        # Preprocess the image
        image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image)
            predicted_digit = torch.argmax(output, 1).item()

        # Return prediction
        return {"predicted_digit": predicted_digit}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
