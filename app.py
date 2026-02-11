import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .emotion-label {
        font-size: 1.4rem;
        font-weight: bold;
    }
    .confidence-text {
        font-size: 1rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Emoji mapping
EMOTION_EMOJIS = {
    "Angry": "😠", "Disgust": "🤢", "Fear": "😨", "Happy": "😊",
    "Neutral": "😐", "Sad": "😢", "Surprise": "😲",
}

EMOTION_COLORS = {
    "Angry": (0, 0, 255), "Disgust": (0, 128, 0), "Fear": (128, 0, 128),
    "Happy": (0, 200, 0), "Neutral": (200, 200, 0), "Sad": (255, 165, 0),
    "Surprise": (255, 255, 0),
}


@st.cache_resource
def load_emotion_model():
    model_path = os.path.join(os.path.dirname(__file__), "model_file_30epochs.h5")
    return load_model(model_path)


@st.cache_resource
def load_face_detector():
    cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    return cv2.CascadeClassifier(cascade_path)


def detect_emotions(image, model, face_detector):
    labels_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

    if isinstance(image, Image.Image):
        frame = np.array(image)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame = image.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 3)

    results = []
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped, verbose=0)
        label_idx = np.argmax(result, axis=1)[0]
        label = labels_dict[label_idx]
        confidence = float(result[0][label_idx]) * 100
        all_probs = {labels_dict[i]: float(result[0][i]) * 100 for i in range(7)}

        color = EMOTION_COLORS.get(label, (50, 50, 255))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
        cv2.putText(frame, f"{label} ({confidence:.1f}%)", (x+5, y-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        results.append({
            "label": label,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "bbox": (x, y, w, h),
        })

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, results


def show_results(image, model, face_detector):
    """Display the detection results."""
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📷 Detection Result")
        with st.spinner("Analyzing emotions..."):
            annotated_image, results = detect_emotions(image, model, face_detector)
        st.image(annotated_image, use_container_width=True)

    with col2:
        st.subheader("📊 Results")
        if len(results) == 0:
            st.warning("No faces detected. Try a different photo with clearly visible faces.", icon="⚠️")
        else:
            st.info(f"**{len(results)} face(s) detected**", icon="👤")
            for i, res in enumerate(results):
                emoji = EMOTION_EMOJIS.get(res["label"], "")
                st.markdown(
                    f'<div class="emotion-card">'
                    f'<div class="emotion-label">{emoji} {res["label"]}</div>'
                    f'<div class="confidence-text">Confidence: {res["confidence"]:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                with st.expander(f"Face {i+1} — Detailed probabilities"):
                    sorted_probs = sorted(res["all_probabilities"].items(), key=lambda x: x[1], reverse=True)
                    for emotion, prob in sorted_probs:
                        e = EMOTION_EMOJIS.get(emotion, "")
                        st.progress(prob / 100, text=f"{e} {emotion}: {prob:.1f}%")


def main():
    st.title("😊 Facial Emotion Recognition")
    st.markdown("Detect emotions from faces using deep learning. Upload an image or take a photo with your webcam.")

    # Load model
    with st.spinner("Loading model..."):
        model = load_emotion_model()
        face_detector = load_face_detector()
    st.success("Model loaded!", icon="✅")

    # Sidebar info
    with st.sidebar:
        st.markdown("### About")
        st.markdown("This app recognizes 7 facial emotions:")
        for emotion, emoji in EMOTION_EMOJIS.items():
            st.markdown(f"- {emoji} **{emotion}**")

    # ---- Two tabs: Upload and Camera ----
    tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Take Photo (Camera)"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload a photo containing one or more faces",
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.markdown("---")
            show_results(image, model, face_detector)

    with tab2:
        st.markdown("Click **Take Photo** below to capture from your webcam.")
        camera_photo = st.camera_input("Take a photo")
        if camera_photo is not None:
            image = Image.open(camera_photo)
            st.markdown("---")
            show_results(image, model, face_detector)


if __name__ == "__main__":
    main()
