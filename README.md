# Deepshield-DFD

An AI-powered deepfake detection system for images and videos. Leverages deep learning and computer vision to identify manipulated content. Supports real-time analysis via a Flask API with GPU acceleration.

## **🚀 Running the Project**

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/neevan0842/Deepshield-DFD.git
cd Deepshield-DFD
```

### **2️⃣ Start the Application**

❗❗Before starting, ensure the command and ports in `docker-compose.yml` is correctly set.

```bash
docker compose up -d

```

### **3️⃣ Stop and Remove Containers**

To stop and remove the running containers, use:

```bash
docker compose down
```

## **🌐 Access the Application**

Once the containers are running, you can access the services at:

- **Flask API**: [http://localhost:5000](http://localhost:5000)
- **Streamlit App**: [http://localhost:8501](http://localhost:8501)

## Usage

#### Upload a video to /uploads and send a POST request to /predict to classify it:

```bash
curl -X POST -F "video=@path_to_video.mp4" http://localhost:5000/predict
```
