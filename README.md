# MediScan-AI — Skin Disease Classification with Explainable AI

[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)]()
[![Medical Imaging](https://img.shields.io/badge/Domain-Dermatology-green)]()

## Overview

**MediScan-AI** is an end-to-end **skin disease classification system** built using deep learning and computer vision techniques. The project focuses on **accurate image-based diagnosis** combined with **model explainability** to improve transparency and trust in medical AI systems.

This repository contains **all implementation files**, including model training, inference, explainability (Grad-CAM), and a **Streamlit-based web interface**.

---

## Problem Motivation

Early detection of skin diseases is critical, yet access to dermatological expertise is limited in many regions. MediScan-AI explores how deep learning models can assist in **automated skin condition recognition** from dermoscopic images.

---

## Dataset

* **HAM10000 (Human Against Machine with 10000 training images)**
* Public dermoscopic image dataset

⚠️ Dataset files are not included due to size and licensing constraints.

---

## Methodology (As Implemented)

### 1. Data Preprocessing

* Image resizing and normalization
* Train–validation split

### 2. Model Architecture

* CNN-based classifier (ResNet backbone)
* Transfer learning using pretrained weights

### 3. Training

* Supervised learning
* Cross-entropy loss
* GPU-accelerated training

### 4. Explainability

* **Grad-CAM** visualization to highlight salient image regions
* Visual explanation of model predictions

### 5. Deployment

* Interactive **Streamlit web application**
* Upload image → prediction → explanation

---

## Repository Structure

```
MediScan-AI/
│── README.md
│── requirements.txt
│── app.py                 # Streamlit application
│── model/
│   └── trained_model.pth
│
│── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── gradcam.py
│
│── results/
│   ├── gradcam_examples.png
│   └── sample_predictions.png
```

---

## Running the Application

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App

```bash
streamlit run app.py
```

---

## Results

* Accurate skin disease classification on HAM10000
* Visual explanations generated via Grad-CAM

Representative results are provided in the `results/` directory.

---

## Limitations

* Research / educational use only
* Not intended for clinical diagnosis
* Dataset-specific evaluation

---

## Why This Project Matters

This project demonstrates:

* Applied deep learning in medical imaging
* Integration of **explainable AI** techniques
* End-to-end ML system development (training → deployment)

It complements research-focused repositories by showcasing **practical deployment skills**.

---

## Author

**Abir Das**
AI Researcher | Medical Imaging & Explainable AI

---

## License

For academic and educational use.
