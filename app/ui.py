import streamlit as st
import os
import torch
import time
from logger import logger
from utils import predict_single_video, load_models, upload_video_streamlit, extract_sample_frames  # Assume extract_sample_frames added

# Paths
PATH_TO_WEIGHTS = os.path.join(os.getcwd(), "app", "weights")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "app", "uploads")

# Check for GPU once
if "gpu_checked" not in st.session_state:
    st.session_state.gpu_checked = True
    if torch.cuda.is_available():
        logger.info("✅ GPU Available")
    else:
        logger.info("❌ GPU Not Available")

# Load Models (cached)
@st.cache_resource
def load_cached_models():
    return load_models(PATH_TO_WEIGHTS)

models = load_cached_models()

# --- 🎨 Custom CSS Styling ---
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stVideo {
        border: 2px solid #0E1117;
        border-radius: 10px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 📚 Sidebar ---
with st.sidebar:
    st.title("DeepShield 🔐")
    st.markdown("👤 Built by **Zeeshan**")
    st.markdown("📧 mohammedzeeshan1704@gmail.com")
    st.markdown("---")
    st.info("💡 Tip: Use short, high-quality videos for best results.")
    st.markdown("---")
    st.markdown("🔬 Powered by PyTorch + Deep Learning")

# --- 🧠 Main App Title ---
st.title("🧠 DeepShield - Deepfake Detection")
st.markdown("Upload a video and our AI will detect if it's **real or fake**.")

# --- 🎥 Upload Video ---
video_file = st.file_uploader("📁 Choose a video file", type=["mp4", "mov", "avi"])

if video_file is not None:
    if (
        "uploaded_video" not in st.session_state
        or st.session_state.uploaded_video != video_file.name
    ):
        st.session_state.uploaded_video = video_file.name
        st.session_state.video_path = upload_video_streamlit(video_file, UPLOAD_FOLDER)
        logger.info(f"Video uploaded: {st.session_state.video_path.split('/')[-1]}")

    # Show uploaded video
    st.video(st.session_state.video_path)

    # Analyze Button
    if st.button("🔍 Analyze Video"):
        with st.spinner("Analyzing video... ⏳"):
            try:
                start = time.time()
                prediction = predict_single_video(
                    video_path=st.session_state.video_path, models=models
                )
                end = time.time()
                logger.info(
                    f"Prediction of {video_file.name} took {end - start:.2f} seconds"
                )

                result = "FAKE" if prediction > 0.5 else "REAL"
                confidence = prediction * 100 if result == "FAKE" else (1 - prediction) * 100

                # 🟩 Result display
                if result == "FAKE":
                    st.error(
                        f"❌ **Prediction:** {prediction:.4f}\n\n⚠️ **Result: FAKE** (Confidence: {confidence:.2f}%)"
                    )
                else:
                    st.success(
                        f"✅ **Prediction:** {prediction:.4f}\n\n🎉 **Result: REAL** (Confidence: {confidence:.2f}%)"
                    )

                # ⏱ Inference Time
                st.write(f"🕒 Inference Time: `{end - start:.2f}` seconds")

                # 📊 Confidence Progress Bar
                st.progress(int(confidence))

                # 📄 Downloadable Report
                report = f"""
                Video: {video_file.name}
                Prediction: {result}
                Confidence: {confidence:.2f}%
                Time Taken: {end - start:.2f} seconds
                """
                st.download_button("📥 Download Prediction Report", report, file_name="deepfake_report.txt")

                # 🖼️ Key Frame Display (Assume util method exists)
                st.markdown("### 🖼️ Sample Analyzed Frames")
                frames = extract_sample_frames(st.session_state.video_path, count=4)
                for i, frame in enumerate(frames):
                    st.image(frame, caption=f"Frame {i+1}", width=150)

            except Exception as e:
                st.error(f"⚠️ **Error:** {e}")
