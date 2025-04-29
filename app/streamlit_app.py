import streamlit as st
import requests
from PIL import Image
import io

# --- Settings ---
st.set_page_config(page_title="MediScan", layout="centered")
st.title("🧠 MediScan – AI Skin Condition Classifier")

st.markdown("""
## 👋 Welcome to MediScan

MediScan is a real-time AI-powered tool that helps identify common skin conditions using deep learning and computer vision.

---

### 🧠 How It Works:
1. 📸 Upload a clear skin image  
2. 🧪 The AI model (running in the cloud) analyzes your image  
3. 📖 Get diagnosis and medical information  
""")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Show uploaded image
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📷 Uploaded Image")
        st.image(image, use_column_width=True)

    # Send image to Flask backend
    st.markdown("### 🔄 Sending image to backend for prediction...")
    files = {"file": uploaded_file}
    try:
        response = requests.post("https://mediscan-flask.onrender.com/predict", files=files)
        if response.status_code == 200:
            result = response.json()
            predicted_class = result["prediction"]
            confidence = result["confidence"]

            with col2:
                st.markdown("### 📊 Prediction")
                st.markdown(f"**🧠 Class:** {predicted_class}")
                st.markdown(f"**📈 Confidence:** {confidence:.2%}")

            # 📖 Condition Info
            disease_key = predicted_class.split("(")[0].strip().lower().replace(" ", "_")
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

> 🔗 **[Learn More on WebMD](https://www.webmd.com/search/search_results/default.aspx?query={predicted_class.split()[0]})**
""")
            except FileNotFoundError:
                st.info("ℹ️ Condition info not found for this class.")

        else:
            st.error(f"❌ Backend error: {response.status_code}")
    except Exception as e:
        st.error(f"⚠️ Could not connect to backend.\n\n{e}")

# Footer
st.markdown("""
---
Built with ❤️ by [Abir Das](https://www.linkedin.com/in/abir-das-0042b1275)  
Powered by Streamlit · Flask · Render · Hugging Face
""")
