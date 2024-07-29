FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the preprocessing script into the container
COPY preprocess_api.py .
COPY model_training_files /app/model_training_files

# Run the script when the container launches
CMD ["python", "preprocess_api.py"]
