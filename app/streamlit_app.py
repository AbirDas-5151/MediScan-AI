# streamlit_app.py

# streamlit_app.py

import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
import torch
from torchvision import transforms
from models.train_model import create_model
from utils.gradcam import apply_gradcam


# --- Settings ---
CLASS_NAMES = [
    "Actinic keratoses (akiec)",
    "Basal cell carcinoma (bcc)",
    "Benign keratosis-like lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanocytic nevi (nv)",
    "Melanoma (mel)",
    "Vascular lesions (vasc)"
]

st.set_page_config(page_title="MediScan", layout="centered")
st.title("🧠 MediScan – AI Skin Condition Classifier")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Layout columns: Image | Grad-CAM
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📷 Uploaded Image")
        st.image(image, use_column_width=True)

    # Load model
    num_classes = len(CLASS_NAMES)
    model = create_model(num_classes)
    model.load_state_dict(torch.load("resnet50_medical.pth", map_location="mps"))

    with st.spinner("🧠 Analyzing with AI..."):
        heatmap_img, pred_class = apply_gradcam(model, uploaded_file)

    with col2:
        st.markdown("### 🌈 Grad-CAM Heatmap")
        st.image(heatmap_img, use_column_width=True)

    st.markdown("---")
    st.markdown(f"## 🧠 Predicted: **{CLASS_NAMES[pred_class]}**")

    # Condition info section
    disease_key = CLASS_NAMES[pred_class].split("(")[0].strip().lower().replace(" ", "_")
    try:
        with open(f"disease_info/{disease_key}.json", "r") as f:
            import json
            disease_info = json.load(f)

        st.markdown("### 📖 Condition Details")
        st.markdown(f"**📝 Description:** {disease_info.get('description', '-')}")
        st.markdown(f"**🧪 Causes:** {', '.join(disease_info.get('causes', []))}")
        st.markdown(f"**🔍 Symptoms:** {', '.join(disease_info.get('symptoms', []))}")
        st.markdown(f"**💊 Treatment:** {', '.join(disease_info.get('treatment', []))}")

    except FileNotFoundError:
        st.info("ℹ️ Condition info not found for this class.")

