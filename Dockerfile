# Use PyTorch base image
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies required for dlib and downloading files
RUN apt-get update && apt-get install -y \
cmake \
gcc \
g++ \
make \
wget \
&& rm -rf /var/lib/apt/lists/*  # Clean up APT cache to reduce image size

# download pretraned Imagenet models
RUN wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth -P /root/.cache/torch/hub/checkpoints/

# upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy only requirements first (caching optimization)
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the project files (AFTER dependencies are installed)
COPY . /workspace

# Ensure download_weights.sh is executable and run it
RUN chmod +x /workspace/download_weights.sh && /workspace/download_weights.sh

# Set environment variables
ENV FLASK_APP=app/main.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV FLASK_DEBUG=1

# Serve Flask API
CMD ["python3", "app/main.py"] 

# Serve Streamlit app
# CMD ["streamlit", "run", "app/ui.py"]
