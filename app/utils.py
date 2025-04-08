import os
import re
import torch
from PIL import Image
import numpy as np
from moviepy.editor import VideoFileClip
from classifiers import DeepFakeClassifier
from kernel_utils import (
    predict_on_video,
    VideoReader,
    FaceExtractor,
    confident_strategy,
)

# ---------------------- Load Models ---------------------- #
def load_models(weights_dir):
    """Load all models from a given weights directory."""
    models = [model for model in os.listdir(weights_dir) if model.startswith("final")]
    loaded_models = []

    for model_name in models:
        path = os.path.join(weights_dir, model_name)
        print(f"Loading state dict from {path}")
        
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Remove 'module.' from keys if present
        cleaned_state_dict = {re.sub("^module.", "", k): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned_state_dict, strict=True)
        model.eval()

        del checkpoint
        loaded_models.append(model.half())

    return loaded_models

# ---------------------- Video Upload ---------------------- #
def upload_video(video, upload_folder):
    """Save uploaded video (from Flask or other backend)."""
    filename = video.filename
    video_path = os.path.join(upload_folder, filename)

    video.save(video_path)
    keep_top_k(upload_folder)

    return video_path

def upload_video_streamlit(video, upload_folder):
    """Save uploaded video in Streamlit environment."""
    filename = video.name
    video_path = os.path.join(upload_folder, filename)

    with open(video_path, "wb") as f:
        f.write(video.read())

    keep_top_k(upload_folder)
    return video_path

def keep_top_k(upload_folder, k=10):
    """Keep only the latest k videos to save space."""
    files = os.listdir(upload_folder)
    files_paths = [os.path.join(upload_folder, f) for f in files if f != ".gitkeep"]
    files_paths.sort(key=lambda x: -os.path.getctime(x))

    while len(files_paths) > k:
        os.remove(files_paths.pop())

# ---------------------- Prediction ---------------------- #
def predict_single_video(video_path, models, frames_per_video=32, input_size=380):
    """Run deepfake prediction on a single video using loaded models."""
    video_reader = VideoReader()
    read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(read_fn)
    strategy = confident_strategy

    prediction = predict_on_video(
        face_extractor=face_extractor,
        video_path=video_path,
        batch_size=frames_per_video,
        input_size=input_size,
        models=models,
        strategy=strategy,
    )

    return float(prediction)

# ---------------------- Frame Extraction ---------------------- #
def extract_sample_frames(video_path, count=4):
    """Extract evenly spaced sample frames from a video."""
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frames = []

    for i in range(count):
        t = (i + 1) * duration / (count + 1)
        frame = clip.get_frame(t)
        image = Image.fromarray(np.uint8(frame))
        frames.append(image)

    return frames
