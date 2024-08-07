# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY . /app

# Install the dependencies
RUN apt-get -y update && apt-get -y install libgl1
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.deepface/weights

RUN python load_weights.py

# Copy the rest of the application code into the container
COPY . .

EXPOSE 8501
# Command to run the application
CMD ["streamlit","run","DeepFaceVid-v2.py", "--server.port=8501","--server.address=0.0.0.0"]
