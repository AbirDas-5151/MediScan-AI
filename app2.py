from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from model_utils import load_model, predict

app = Flask(__name__)
model = load_model()

@app.route("/")
def home():
    return "🧠 MediScan Flask API is running!"

@app.route("/predict", methods=["POST"])
def handle_prediction():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")

    prediction, confidence = predict(model, image)
    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    app.run(debug=True)
