![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)




# 🫁 PneumoScan — Chest X-Ray Pneumonia Detector

A custom CNN trained from scratch to classify chest X-rays as **Pneumonia** or **Normal**, deployed with a Streamlit interface styled like a radiology light-box viewer.
LIVE APP:https://chest-x-ray-pneumonia-detector.streamlit.app/
## Overview
- Input: chest X-ray image (JPG/PNG)
- Output: Pneumonia / No Pneumonia + confidence score
- Model: Convolutional Neural Network (Keras/TensorFlow)
- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Model Architecture
- Input: 128×128 grayscale
- 4 Conv blocks (32 → 64 → 128 → 256 filters, 3 Conv2D layers each) + MaxPooling
- Flatten → Dense(256) → BatchNormalization → Dense(1, sigmoid)
- Loss: Binary Crossentropy | Optimizer: Adam (lr=0.001)

## Tech Stack
- TensorFlow / Keras
- Streamlit
- NumPy, Pillow

## Preprocessing Pipeline
- Grayscale conversion
- Resize to 128×128
- Normalization (pixel values / 255.0)

## Disclaimer
⚕️ This is a student ML project for educational purposes only — **not** a substitute for professional medical diagnosis.

## Author
Fawad Ahmad Bilal — [GitHub](https://github.com/FawadAhmad-bilal)
