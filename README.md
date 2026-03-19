
# 🩺 Chest X-ray Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red)](https://streamlit.io/)
[![Medical AI](https://img.shields.io/badge/Domain-Medical%20Imaging-purple)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![GradCAM](https://img.shields.io/badge/Explainability-GradCAM-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-success?logo=streamlit)](https://chest-x-ray-disease-detection-system-by-rohit-raut.streamlit.app/)

---

## 📌 Project Overview

This project demonstrates an **end-to-end Deep Learning system for automated chest X-ray disease detection**.

The system uses a **DenseNet121 Convolutional Neural Network (CNN)** trained using **transfer learning** on the **NIH Chest X-ray dataset** to detect **14 thoracic diseases**.

### The project includes:

- **Deep Learning model training**
- **Medical image preprocessing**
- **Multi-label disease classification**
- **Explainable AI using Grad-CAM**
- **Interactive Streamlit diagnostic dashboard**
- **PDF diagnostic report generation**
- **Live deployment on Streamlit Community Cloud**

---

## 🌐 Live Demo

🔗 **Try the deployed application**

https://chest-x-ray-disease-detection-system-by-rohit-raut.streamlit.app/

This interactive application allows users to:

- Upload chest X-ray images
- Run AI prediction
- View disease probabilities
- Visualize **GradCAM attention maps**
- Download an **AI diagnostic report (PDF)**

---

## 🧠 Problem Statement

Chest X-ray interpretation is one of the most common diagnostic procedures in **medical imaging**.

However:

- Radiologists often face **large workloads**
- Subtle disease patterns can be **difficult to detect**

Deep Learning models can assist clinicians by providing **automated disease detection** and **visual explanations**.

This project demonstrates how **Artificial Intelligence can support medical imaging analysis**.

---

## 🗂 Dataset Information

**Dataset Source**

NIH Chest X-ray Dataset
https://www.kaggle.com/datasets/nih-chest-xrays/data

### Dataset Characteristics

| Property | Value |
|--------|------|
| Dataset Type | Medical Imaging |
| Modality | Chest Radiographs |
| Image Type | Frontal Chest X-rays |
| Total Images Used | ~50,000 |
| Disease Labels | 14 |

---

## 🦠 Disease Classes

The model predicts the following **14 thoracic diseases**:

1. **Atelectasis**
2. **Cardiomegaly**
3. **Effusion**
4. **Infiltration**
5. **Mass**
6. **Nodule**
7. **Pneumonia**
8. **Pneumothorax**
9. **Consolidation**
10. **Edema**
11. **Emphysema**
12. **Fibrosis**
13. **Pleural Thickening**
14. **Hernia**

---

## ⚙️ Deep Learning Pipeline

The project follows a **complete deep learning workflow**:

```bash
          Raw Chest X-ray Dataset
                    ↓
            Data Preprocessing
                    ↓
        Label Encoding (Multi-label)
                    ↓
       Image Normalization & Resizing
                    ↓
       Transfer Learning (DenseNet121)
                    ↓
              Model Training
                    ↓
             Model Evaluation
                    ↓
          GradCAM Explainability
                    ↓
        Streamlit Web Application
                    ↓
             Cloud Deployment
```

---

## 🧹 Data Preprocessing

The dataset undergoes several preprocessing steps:

### Image Processing

- Images resized to **224 × 224**
- Pixel values normalized to **[0,1]**
- Converted to **RGB format**

### Label Encoding

The dataset contains **multi-label annotations**.

**Example label:**

```bash
Atelectasis | Effusion
```

Converted to **multi-hot encoded vector:**

```bash
[1,0,1,0,0,0,0,0,0,0,0,0,0,0]
```

---

## 🧠 Model Architecture

The system uses **Transfer Learning with DenseNet121**.

### Architecture

```bash
    Input Image (224 × 224 × 3)
                ↓
  DenseNet121 (ImageNet Pretrained)
                ↓
       Global Average Pooling
                ↓
        Batch Normalization
                ↓
             Dropout
                ↓
  Dense Layer (14 neurons, Sigmoid)
                ↓
   Multi-Label Disease Predictions
```

### Why DenseNet121?

- Strong performance in **medical imaging**
- Efficient parameter usage
- Dense connectivity improves **gradient flow**
- Proven results on **chest X-ray classification tasks**

---

## 🏋️ Model Training

### Training Environment

| Parameter | Value |
|-----------|-------|
| Framework | TensorFlow / Keras |
| Hardware | NVIDIA Tesla T4 GPU |
| Epochs | 10 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |

---

## 📈 Model Performance

Final results after training:

| Metric | Score |
|------|------|
| Training AUC | 0.84 |
| Validation AUC | 0.79 |
| Training Loss | 0.157 |
| Validation Loss | 0.172 |

These results demonstrate that the model learns **meaningful medical image features** while maintaining **reasonable generalization**.

---

## 🔎 Explainable AI (Grad-CAM)

Medical AI requires **interpretability**.

This project uses **GradCAM (Gradient-weighted Class Activation Mapping)**.

GradCAM highlights the **regions of the chest X-ray that influenced the model's prediction**.

### Workflow

```bash
     Chest X-ray Image
            ↓
      Model Prediction
            ↓
  Gradient Backpropagation
            ↓
    Heatmap Generation
            ↓
 Highlighted Disease Regions
```

This improves **model transparency and trust**.

---

## 🖥️ Streamlit Dashboard

The trained model is deployed as an **interactive medical AI application**.

### Features

- X-ray image upload
- AI disease prediction
- Confidence gauge visualization
- Probability charts
- GradCAM heatmap visualization
- AI diagnostic PDF report

---

## 📂 Project Structure

```bash
chest-x-ray-disease-detection
│
├── streamlit_app/
│   └── app.py
│
├── saved_models/
│   └── chest_xray_model.keras
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── dataset_loader.py
│   ├── explainability/
│   │   └── gradcam.py
│   ├── models/
│   │   └── build_model.py
│   ├── preprocessing/
│   │   ├── image_loader.py
│   │   └── label_encoder.py
│   └── training/
│       └── train_model.py
│
├── notebooks/
│   ├── data_exploration_eda.ipynb
│   ├── test_gradcam.ipynb
│   └── test_model.ipynb
│
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
├── README.md
└── .gitignore
```

---

## 🛠 Tech Stack

### Programming
- Python

### Deep Learning
- TensorFlow / Keras

### Data Processing
- NumPy
- Pandas

### Visualization
- Plotly
- Matplotlib

### Image Processing
- OpenCV
- Pillow

### Explainable AI
- GradCAM

### Web Application
- Streamlit

---

## 🚀 Run Locally

### Clone repository
```bash
git clone https://github.com/rohit-1024/chest-x-ray-disease-detection.git
```

### Navigate to project
```bash
cd chest-x-ray-disease-detection
```

### Create virtual environment
```bash
python -m venv .venv
```

### Activate environment
```bash
 .\.venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements-dev.txt
pip install -r requirements.txt
```

### Run Streamlit app
```bash
streamlit run streamlit_app/app.py
```

---

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

### Deployment Steps

1. Push project to **GitHub**
2. Connect repository with **Streamlit Cloud**
3. Set entry point

```bash
streamlit_app/app.py
```

---

## ⚠️ Medical Disclaimer

This project is intended for **educational and research purposes only**.

The predictions generated by this system **must not be used for clinical diagnosis**.

Always consult **qualified medical professionals** for medical decisions.

---

## 🤝 Contributions

Contributions are welcome!

Feel free to **fork the repository** and submit **pull requests**.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Rohit Raut**

📧 Email: rohit.it4368@gmail.com

🐙 GitHub: https://github.com/rohit-1024

💼 LinkedIn: https://www.linkedin.com/in/rohit1024/

---
