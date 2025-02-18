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

# Copy only requirements first (caching optimization)
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy the rest of the project files (AFTER dependencies are installed)
COPY . /workspace

# Ensure download_weights.sh is executable and run it
# RUN chmod +x /workspace/download_weights.sh && /workspace/download_weights.sh

# Expose necessary port (for Flask API and streamlit app)
EXPOSE 5000
EXPOSE 8501

# Serve Flask API
# CMD ["python3", "app/main.py"] 

# Serve Streamlit app
CMD ["streamlit", "run", "app/ui.py"]
