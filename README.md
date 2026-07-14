# 🌸 Flower AI Classifier using Convolutional Neural Networks (CNN)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b?logo=streamlit)

## 📌 Project Overview

This project is a **Deep Learning-based Flower Image Classification System** developed using **TensorFlow, Keras, and Streamlit**. It classifies flower images into one of five categories using a custom Convolutional Neural Network (CNN).

The project demonstrates the complete deep learning workflow, including dataset preprocessing, model training, evaluation, saving the trained model, and deploying it as an interactive web application.

---

## 🌼 Flower Classes

The model classifies images into the following categories:

- 🌼 Daisy
- 🌿 Dandelion
- 🌹 Roses
- 🌻 Sunflowers
- 🌷 Tulips

---

## 🚀 Features

- Image classification using a custom CNN
- TensorFlow Flowers dataset
- Data augmentation for improved generalization
- Streamlit-based interactive web application
- Displays predicted flower class
- Displays prediction confidence
- Shows confidence scores for all flower classes
- Modern and user-friendly interface

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow
- Streamlit

---

## 📂 Project Structure

```
DeepVision-CNN/
│
├── app.py
├── README.md
├── requirements.txt
│
├── assets/
│   └── flowerbanner.jpg
│
├── models/
│   └── flower_cnn_model.keras
│
├── notebooks/
│   └── ImageClassification.ipynb
│
├── outputs/
│   ├── accuracy.png
│   └── loss.png
│
└── src/
    ├── model.py
    ├── train.py
    ├── predict.py
    └── utils.py
```

---

## 📊 Model Performance

| Metric | Value |
|---------|------:|
| Validation Accuracy | **~77.9%** |
| Classes | 5 |
| Image Size | 180 × 180 |
| Framework | TensorFlow / Keras |

---

## 📈 Training Results

### Accuracy Curve

![Accuracy](outputs/accuracy.png)

### Loss Curve

![Loss](outputs/loss.png)

---

## 🖥️ Streamlit Application

The application allows users to:

- Upload a flower image
- Predict the flower species
- View prediction confidence
- Compare confidence scores across all flower classes

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_LINK>
```

Navigate to the project:

```bash
cd DeepVision-CNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📷 Example Workflow

1. Launch the Streamlit application.
2. Upload a flower image.
3. The model predicts the flower species.
4. Confidence scores for all classes are displayed.

---

## 📚 Dataset

This project uses the **TensorFlow Flowers Dataset**, which contains approximately **3,670 flower images** across five categories.

---

## 🎯 Future Improvements

- Improve classification accuracy using Transfer Learning (MobileNetV2, EfficientNet, or ResNet50)
- Add Grad-CAM visualizations for model explainability
- Deploy the application to Streamlit Community Cloud
- Support additional flower species
- Improve inference speed and optimization

---

## 👩‍💻 Author

**Syeda Momina Sherazi**

BS Software Engineering  
National University of Sciences and Technology (NUST)

---

## ⭐ Acknowledgements

- TensorFlow
- Keras
- Streamlit
- TensorFlow Flowers Dataset