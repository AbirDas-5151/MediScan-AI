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
st.markdown("""
## 👋 Welcome to MediScan

MediScan is a real-time AI-powered tool that helps identify common skin conditions using deep learning and computer vision.

---

### 🧠 How It Works:
1. 📸 Upload a clear skin image
2. 🧪 The AI model analyzes your image
3. 🌈 A Grad-CAM heatmap highlights the focus area
4. 📖 Get details about the predicted condition
""")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Layout columns: Image | Grad-CAM
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📷 Uploaded Image")
        st.image(image, use_container_width=True)

    # Load model
    num_classes = len(CLASS_NAMES)
    model = create_model(num_classes)
    model.load_state_dict(torch.load("resnet50_medical.pth", map_location="mps"))

    with st.spinner("🧠 Analyzing with AI..."):
        heatmap_img, pred_class = apply_gradcam(model, uploaded_file)

    with col2:
        st.markdown("### 🌈 Grad-CAM Heatmap")
        st.image(heatmap_img, use_container_width=True)

    st.markdown("---")
    st.markdown(f"## 🧠 Predicted: **{CLASS_NAMES[pred_class]}**")

    # 🔢 Confidence Chart
    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0).to("mps", dtype=torch.float32)
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()

    def plot_class_probabilities(probs, class_names):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(class_names, probs, color='skyblue')
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("Confidence")
        ax.set_title("Prediction Confidence per Class")
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{width:.2f}", va='center')
        return fig

    st.markdown("### 📊 Prediction Confidence")
    st.pyplot(plot_class_probabilities(probs, CLASS_NAMES))

    # 📖 Condition info section
    disease_key = CLASS_NAMES[pred_class].split("(")[0].strip().lower().replace(" ", "_")
    try:
        with open(f"disease_info/{disease_key}.json", "r") as f:
            import json
            disease_info = json.load(f)

        st.markdown("### 📖 Condition Details")
        st.markdown(f"""
- **📝 Description**: {disease_info.get('description', '-')}
- **🧪 Causes**: {', '.join(disease_info.get('causes', []))}
- **🔍 Symptoms**: {', '.join(disease_info.get('symptoms', []))}
- **💊 Treatment**: {', '.join(disease_info.get('treatment', []))}

> 🔗 **[Learn More on WebMD](https://www.webmd.com/search/search_results/default.aspx?query={CLASS_NAMES[pred_class].split()[0]})**
""")

    except FileNotFoundError:
        st.info("ℹ️ Condition info not found for this class.")

# Footer
st.markdown("""
---
Built with ❤️ by [Abir Das](https://www.linkedin.com/in/abir-das-0042b1275)  
Powered by Streamlit · PyTorch · Hugging Face
""")
