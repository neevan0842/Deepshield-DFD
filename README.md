# Deepshield-DFD

AI-powered deepfake detection for images and videos. Uses deep learning and computer vision to identify manipulated content. Supports real-time analysis via Flask API with GPU acceleration.

## Setup

1. Clone the repo:

```bash
git clone https://github.com/neevan0842/Deepshield-DFD.git
cd DeepShield-DFD
```

2. Download pre-trained weights:

```bash
bash download_weights.sh
```

3. Build Docker image:

```bash
docker build -t deepshield .
```

4. Run the Docker container:

```bash
docker run --gpus all -d -p 5000:5000 -p 8501:8501 -v .:/workspace --name deepshield deepshield
```

## Usage

#### Upload a video to /uploads and send a POST request to /predict to classify it:

```bash
curl -X POST -F "video=@path_to_video.mp4" http://localhost:5000/predict
```
