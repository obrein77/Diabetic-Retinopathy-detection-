import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from huggingface_hub import hf_hub_download

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="APTOS DR Detection",
    page_icon="🔬",
    layout="centered"
)

# ── LOAD MODEL FROM HUGGING FACE ─────────────────────────────
@st.cache_resource
def load_dr_model():
    st.info("Loading model... please wait")
    model_path = hf_hub_download(
        repo_id="kithinjibrian95/aptos-resnet50",
        filename="resnet50_aptos_best.keras"
    )
    model = load_model(model_path, compile=False, safe_mode=False)
    return model

# ── LABELS ───────────────────────────────────────────────────
LABELS = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

DESCRIPTIONS = {
    "No DR": "No signs of diabetic retinopathy detected.",
    "Mild DR": "Mild non-proliferative DR. Small microaneurysms present.",
    "Moderate DR": "Moderate non-proliferative DR. More widespread changes.",
    "Severe DR": "Severe non-proliferative DR. Urgent medical attention needed.",
    "Proliferative DR": "Proliferative DR. Advanced stage. Immediate treatment required."
}

COLORS = {
    "No DR": "🟢",
    "Mild DR": "🟡",
    "Moderate DR": "🟠",
    "Severe DR": "🔴",
    "Proliferative DR": "🚨"
}

# ── PREDICTION FUNCTION ───────────────────────────────────────
def predict(img):
    img = img.resize((224, 224))
    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array.astype(np.float32), axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    model = load_dr_model()
    predictions = model.predict(img_array, verbose=0)
    result = np.argmax(predictions[0])
    confidence = predictions[0][result] * 100

    return LABELS[result], confidence, predictions[0]

# ── UI ───────────────────────────────────────────────────────
st.title("🔬 Diabetic Retinopathy Detection")
st.subheader("ResNet50 Transfer Learning | APTOS 2019 Dataset")
st.markdown("Upload a retinal fundus image to detect DR severity.")
st.markdown("---")

uploaded_file = st.file_uploader(
    "📤 Upload Retinal Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Retinal Image", use_container_width=True)

    with col2:
        with st.spinner("🔍 Analyzing image..."):
            pred_name, confidence, all_probs = predict(img)

        st.markdown(f"### {COLORS[pred_name]} {pred_name}")
        st.markdown(f"**Confidence:** {confidence:.1f}%")
        st.info(DESCRIPTIONS[pred_name])

    st.markdown("---")
    st.markdown("### 📊 All Class Probabilities")

    for i, label in enumerate(LABELS):
        prob = float(all_probs[i]) * 100
        st.progress(
            int(prob),
            text=f"{COLORS[label]} {label}: {prob:.1f}%"
        )

    st.markdown("---")
    st.warning("⚠️ AI screening tool only. Consult a qualified ophthalmologist for medical diagnosis.")

else:
    st.info("👆 Please upload a retinal image to get started")

# ── ABOUT ────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    | Detail | Value |
    |--------|-------|
    | Architecture | ResNet50 + Transfer Learning |
    | Dataset | APTOS 2019 (3,662 images) |
    | Validation Accuracy | ~81% |
    | Classes | 5 DR severity levels |
    | Training Platform | Modal.com + Tesla T4 GPU |
    | Built by | Obrien — EEE, Dedan Kimathi University |
    """)
