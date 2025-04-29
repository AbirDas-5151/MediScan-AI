import streamlit as st
import requests
import json
import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
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

# Load model locally for Grad-CAM + chart
from models.train_model import create_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("resnet50_medical_cpu.pth", map_location=device))
model.to(device)
model.eval()

# Streamlit UI
st.set_page_config(page_title="MediScan", layout="centered")
st.title("🧠 MediScan – AI Skin Condition Classifier")

st.markdown("""
## 👋 Welcome to MediScan

MediScan is a real-time AI-powered tool that helps identify common skin conditions using deep learning and computer vision.

---

### 🧠 How It Works:
1. 📸 Upload a clear skin image  
2. 🧪 AI model predicts disease  
3. 🌈 Grad-CAM shows focus area  
4. 📖 Learn condition details
""")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📷 Uploaded Image")
        st.image(image, use_container_width=True)

    # Send image to Flask API
    files = {"file": uploaded_file}
    response = requests.post("https://your-flask-url.onrender.com/predict", files=files)

    if response.status_code == 200:
        result = response.json()
        predicted_class = result["prediction"]
        confidence = result["confidence"]

        # Get class index (needed for Grad-CAM)
        class_idx = CLASS_NAMES.index(predicted_class)

        # 🔥 Grad-CAM (locally)
        heatmap_img, _ = apply_gradcam(model, uploaded_file, class_idx=class_idx)

        with col2:
            st.markdown("### 🌈 Grad-CAM Heatmap")
            st.image(heatmap_img, use_container_width=True)

        # Show prediction + confidence
        st.markdown("---")
        st.markdown(f"## 🧠 Predicted: **{predicted_class}**")
        st.markdown(f"**📈 Confidence:** {confidence:.2%}")

        # 📊 Chart (locally)
        with torch.no_grad():
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)
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

        # 📖 Condition Info
        disease_key = predicted_class.split("(")[0].strip().lower().replace(" ", "_")
        try:
            with open(f"disease_info/{disease_key}.json", "r") as f:
                disease_info = json.load(f)

            st.markdown("### 📖 Condition Details")
            st.markdown(f"""
- **📝 Description**: {disease_info.get('description', '-')}
- **🧪 Causes**: {', '.join(disease_info.get('causes', []))}
- **🔍 Symptoms**: {', '.join(disease_info.get('symptoms', []))}
- **💊 Treatment**: {', '.join(disease_info.get('treatment', []))}

> 🔗 **[Learn More on WebMD](https://www.webmd.com/search/search_results/default.aspx?query={predicted_class.split()[0]})**
""")
        except FileNotFoundError:
            st.info("ℹ️ Condition info not found for this class.")
    else:
        st.error("❌ Failed to connect to backend.")

# Footer
st.markdown("""
---
Built with ❤️ by [Abir Das](https://www.linkedin.com/in/abir-das-0042b1275)  
Powered by Streamlit · PyTorch · Flask · Hugging Face · Render
""")
