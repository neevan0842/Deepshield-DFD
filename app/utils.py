import os
import torch
import re
from classifiers import DeepFakeClassifier
from kernel_utils import (
    predict_on_video,
    VideoReader,
    FaceExtractor,
    confident_strategy,
)


def load_models(weights_dir):
    models = [model for model in os.listdir(weights_dir) if model.startswith("final")]
    loaded_models = []
    model_paths = [os.path.join(weights_dir, model) for model in models]
    for path in model_paths:
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
        print(f"Loading state dict {path}")
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(
            {re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True
        )
        model.eval()
        del checkpoint
        loaded_models.append(model.half())
    return loaded_models


def upload_video(video, upload_folder):
    filename = video.filename
    video_path = os.path.join(upload_folder, filename)

    # Save the new video
    video.save(video_path)

    # Get list of files in the folder
    files = os.listdir(upload_folder)
    files_paths = [os.path.join(upload_folder, f) for f in files]

    # Check if there are more than 10 files
    if len(files_paths) > 10:
        # Sort files by creation time (oldest first)
        files_paths.sort(key=lambda x: os.path.getctime(x))

        # Delete the oldest file
        oldest_file = files_paths[0]
        os.remove(oldest_file)

    return video_path


def predict_single_video(video_path, models, frames_per_video=32, input_size=380):
    # Initialize utilities for prediction
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn)
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
