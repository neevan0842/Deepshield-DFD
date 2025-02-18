from flask import Flask, request, jsonify
from utils import load_models, upload_video, predict_single_video
import os
import torch

PATH_TO_WEIGHTS = os.path.join(os.getcwd(), "app", "weights")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "app", "uploads")

if torch.cuda.is_available():
    print("GPU Available")
else:
    print("GPU Not Available")


models = load_models(PATH_TO_WEIGHTS)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)


@app.route("/")
def root():
    return jsonify({"message": "Hello, World!"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]

    if video.filename == "":
        return jsonify({"error": "No video file provided"}), 400

    video_path = upload_video(video, UPLOAD_FOLDER)

    prediction = predict_single_video(video_path=video_path, models=models)
    result = "REAL" if prediction <= 0.5 else "FAKE"

    return jsonify({"prediction": prediction, "result": result}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
