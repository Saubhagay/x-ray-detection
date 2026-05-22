---
title: Chest X-Ray AI Detector
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Chest X-Ray AI Detector

A robust Deep Learning pipeline that classifies Chest X-ray images into three categories: **Normal**, **Pneumonia**, and **Tuberculosis**. It uses the **DenseNet121** architecture (Transfer Learning) to achieve high accuracy and includes Grad-CAM visualization to explain its predictions.

---

## 1. Environment Setup

Make sure you have Python installed, then install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Dataset Structure

To train this model, you need chest X-ray images. We recommend using merged Kaggle datasets to save time. Ensure your dataset is extracted and placed directly into a folder named `dataset` in the same directory as these scripts.

The structure **must** look exactly like this:
```text
x-ray-detector/
├── dataset/
│   ├── train/
│   │   ├── Normal/
│   │   ├── Pneumonia/
│   │   └── Tuberculosis/
│   ├── val/
│   │   ├── ...
│   └── test/
│       ├── ...
```

---

## 3. Model Training
Once your data is ready, train the DenseNet121 model. It uses Early Stopping to prevent overfitting and automatically saves the best version.

```bash
python train.py
```
*(Tip: Deep learning on large images is computationally heavy. If you don't have an NVIDIA GPU, I highly recommend running `train.py` in Google Colab, downloading the resulting `.keras` file, and placing it in a `saved_models` folder here.)*

---

## 4. Evaluation
Check your model's real-world accuracy on the unseen test data.

```bash
python evaluate.py
```
This will generate:
1.  **Test Accuracy** & **AUC Score**
2.  **Classification Report** (Precision, Recall, F1-Score)
3.  A **Confusion Matrix** image saved in the `results/` folder.

---

## 5. Real-Time Prediction & Grad-CAM Visualization
**This is the main tool you asked for.** Pass any X-ray image to this script. It will output a highly accurate diagnosis and show you a heatmap of exactly *where* the model is looking in the lung to make its decision.

```bash
python predict.py "path\to\some_xray_image.jpg"
```
