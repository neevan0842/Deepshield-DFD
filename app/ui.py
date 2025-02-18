import streamlit as st
import os
import torch
from utils import predict_single_video, load_models, upload_video_streamlit

if torch.cuda.is_available():
    print("GPU Available")
else:
    print("GPU Not Available")

PATH_TO_WEIGHTS = os.path.join(os.getcwd(), "app", "weights")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "app", "uploads")

models = load_models(PATH_TO_WEIGHTS)

# Streamlit UI
st.title("DeepShield - Deepfake Detection")
st.markdown("Upload a video to check if it's real or fake.")

# Upload video
video_file = st.file_uploader(
    "Choose a video...", type=["mp4", "mov", "avi"]
)  # TODO: not sure these extension will work

if video_file is not None:
    # Save uploaded video
    video_path = upload_video_streamlit(video_file, UPLOAD_FOLDER)

    # Display video
    st.video(video_path)

    # Analyze button
    if st.button("🔍 Analyze Video"):
        with st.spinner("Analyzing... ⏳"):
            try:
                prediction = predict_single_video(video_path=video_path, models=models)
                result = "FAKE" if prediction > 0.5 else "REAL"
                # Confidence
                confidence = (
                    prediction * 100 if result == "FAKE" else (1 - prediction) * 100
                )

                # Display the result
                if result == "FAKE":
                    st.error(
                        f"❌ **Prediction:** {prediction:.4f}\n\n⚠️ **Result: FAKE** (Confidence: {confidence:.2f}%)"
                    )
                else:
                    st.success(
                        f"✅ **Prediction:** {prediction:.4f}\n\n🎉 **Result: REAL** (Confidence: {confidence:.2f}%)"
                    )

            except Exception as e:
                st.error(f"⚠️ **Error:** {e}")
