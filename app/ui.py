import streamlit as st
import os
import torch
import time
from logger import logger
from utils import predict_single_video, load_models, upload_video_streamlit

# Paths
PATH_TO_WEIGHTS = os.path.join(os.getcwd(), "app", "weights")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "app", "uploads")


if "gpu_checked" not in st.session_state:
    st.session_state.gpu_checked = True
    if torch.cuda.is_available():
        logger.info("GPU Available")
    else:
        logger.info("GPU Not Available")


@st.cache_resource
def load_cached_models():
    return load_models(PATH_TO_WEIGHTS)


models = load_cached_models()


# Streamlit UI
st.title("DeepShield - Deepfake Detection")
st.markdown("Upload a video to check if it's real or fake.")

# Upload video
video_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

if video_file is not None:
    if (
        "uploaded_video" not in st.session_state
        or st.session_state.uploaded_video != video_file.name
    ):
        st.session_state.uploaded_video = video_file.name
        st.session_state.video_path = upload_video_streamlit(video_file, UPLOAD_FOLDER)
        logger.info(f"Video uploaded: {st.session_state.video_path.split('/')[-1]}")

    # Display video
    st.video(st.session_state.video_path)

    # Analyze button
    if st.button("🔍 Analyze Video"):
        with st.spinner("Analyzing... ⏳"):
            try:
                start = time.time()
                prediction = predict_single_video(
                    video_path=st.session_state.video_path, models=models
                )
                end = time.time()
                logger.info(f"Prediction took {end - start:.2f} seconds")
                result = "FAKE" if prediction > 0.5 else "REAL"
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
