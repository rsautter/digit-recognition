from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import logging
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import io
import numpy as np

import onnxruntime as ort
import onnx
#from onnx2pytorch import ConvertModel



# Configure a handler to output logs to the console (optional)
logger = logging.getLogger("fastapi")
logger.setLevel(logging.DEBUG)  # Or logging.TRACE for even more detail
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)  # Or logging.TRACE
logger.addHandler(handler)



onnx_model = onnx.load('rede.onnx')

ort_session = ort.InferenceSession('rede.onnx')
input_name = ort_session.get_inputs()[0].name  # Get the input name
output_name = ort_session.get_outputs()[0].name # Get the output name



# Define transformation for input
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
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
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")

        # Check if the image is blank
        if np.array(image).sum() == 255 * image.size[0] * image.size[1]:
            return JSONResponse(content={"error": "Image is blank. Please provide a valid digit."}, status_code=400)

        # Preprocess the image
        image = transform(image)

        # Make prediction
        try:
            image_np = image.unsqueeze(0).numpy() # Convert to NumPy with batch dimension
            ort_inputs = {input_name: image_np}
            ort_outputs = ort_session.run([output_name], ort_inputs)
            output = torch.from_numpy(ort_outputs[0])  # Convert back to PyTorch tensor

            predicted_digit = torch.argmax(output, 1).item()
        except FileNotFoundError:
            logger.exception("error"+str(e))
            return JSONResponse(content={"error": "Image file not found"}, status_code=400)
        except Exception as e:
            logger.exception("error"+str(e))
            return JSONResponse(content={"error": str(e)}, status_code=500)

        # Return prediction
        return {"predicted_digit": predicted_digit}

    except Exception as e:
        logger.exception("error"+str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.exception_handler(Exception)  # Global exception handler
async def generic_exception_handler(request, exc):
    import traceback
    traceback.print_exc()  # Print the full traceback to the console
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}, # Return the error message in the response
    )
