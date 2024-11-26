import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
from streamlit_drawable_canvas import st_canvas

# Streamlit app setup
st.title("Digit Recognizer")
st.write("Draw a digit (0-9) below, and the model will predict it!")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="rgb(255, 255, 255)",  # White background
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# If the user draws something, process it
if canvas_result.image_data is not None:
    # Convert the canvas image to grayscale
    image = Image.fromarray(np.uint8(canvas_result.image_data)).convert("L")
    
    # Check if the image is blank
    if np.array(image).sum() == 255 * image.size[0] * image.size[1]:
        st.write("### Draw a digit to make a prediction!")
    else:
        # Save the image to a buffer
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        # Send the image to FastAPI for prediction
        response = requests.post("http://127.0.0.1:8000/predict/", files={"file": buffer})
        
        # Display the prediction
        if response.status_code == 200:
            result = response.json()
            st.write(f"### Predicted Digit: {result['predicted_digit']}")
        else:
            st.write("### Error: Unable to process image.")

