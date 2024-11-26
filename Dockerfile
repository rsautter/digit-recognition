# Step 1: Use the official Python runtime as the base image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements.txt file first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy the entire app directory into the container (the contents of ./app)
COPY ./app /app

# Step 5: Expose necessary ports
EXPOSE 8000 8501

# Step 6: Command to run both FastAPI and Streamlit concurrently
CMD ["sh", "-c", "uvicorn backendapi:app --host 0.0.0.0 --port 8000 & streamlit run /app/app.py --server.port=8501 --server.address=0.0.0.0"]
