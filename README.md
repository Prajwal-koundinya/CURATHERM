
# **Curatherm: Breast Cancer Detection Using Infrared Thermography**

### 🌟 **Project Overview**

**Curatherm** is an innovative approach to breast cancer detection leveraging **infrared thermography** and **machine learning models**. This project aims to provide a non-invasive, affordable, and efficient method for early cancer detection. The accompanying app ensures a user-friendly experience while fostering motivation and positivity during the diagnostic process.

---

## 🌐 **Repository Structure**

```
Curatherm/
├── data/
│   ├── benign/
│   ├── malignant/
│   └── preprocessed/
├── models/
│   ├── cnn_model.h5
│   └── thermography_classifier.ipynb
├── app/
│   ├── frontend/
│   │   ├── assets/
│   │   └── components/
│   └── backend/
│       ├── app.py
│       └── utils.py
├── reports/
│   ├── performance_metrics.pdf
│   └── visualizations/
├── README.md
└── requirements.txt
```

### **Main Components**
1. **`data/`**: Contains thermal images categorized into benign and malignant classes, along with preprocessed images.
2. **`models/`**: Houses the machine learning model and related scripts.
3. **`app/`**: Source code for the Curatherm app (frontend and backend).
4. **`reports/`**: Evaluation metrics, visualizations, and analysis reports.

---

## 🎯 **Objectives**

- To develop a machine learning model for classifying breast thermography images into **benign** or **malignant** categories.
- To preprocess thermal images for feature extraction and enhance classification accuracy.
- To design a motivational and responsive **mobile application** for seamless user interaction.
- To evaluate the model's performance and optimize it for real-world applications.

---

## 🚀 **Features**

1. **Cutting-Edge Technology**: Utilizes thermal imaging and CNN-based models for detection.
2. **Non-Invasive**: Provides a painless and safe alternative to traditional diagnostic methods.
3. **Accessible**: Affordable and deployable on mobile devices.
4. **User-Centric Design**: Aims to reduce patient anxiety with a positive and intuitive interface.
5. **Scalable**: Can be expanded to detect other types of cancer or medical anomalies.

---

## ⚙️ **Installation and Usage**

### Prerequisites

- **Python 3.8+**
- **TensorFlow 2.x**
- **OpenCV**
- **Flask/Django (Backend)**

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Curatherm.git
   cd Curatherm
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**:
   Navigate to the `app/backend/` directory and execute:
   ```bash
   python app.py
   ```

4. **Access the Frontend**:
   Open your browser and navigate to `http://localhost:5000`.

---

## 🧠 **Machine Learning Workflow**

### 1. **Data Collection**
   - Dataset consists of thermal images categorized as benign and malignant.
   - Images sourced from clinical trials and publicly available repositories.

### 2. **Preprocessing**
   - Convert thermal images to grayscale.
   - Resize and normalize images.
   - Extract critical features using advanced techniques.

### 3. **Model Training**
   - Model: Convolutional Neural Network (CNN).
   - Framework: TensorFlow/Keras.
   - Training on an 80/20 train-test split.

### 4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score, ROC Curve.
   - **Achieved Accuracy**: ~94% on test data.

---

## 📊 **Performance Metrics**

- **Accuracy**: 94.2%
- **Precision**: 91.8%
- **Recall**: 92.5%
- **F1-Score**: 92.1%

### 📈 Visualization of ROC Curve

![ROC Curve](link-to-roc-curve.png)

---

## 📱 **App Features**

### **Home Screen**
- Interactive UI with clear instructions for users.
- Positive messages to encourage and reassure patients.

### **Upload and Analyze**
- Users can upload thermography images.
- Instant analysis with results displayed alongside confidence scores.

### **Motivational Support**
- Provides motivational quotes and success stories to uplift users.

---

## 🔍 **Future Scope**

1. Enhance model robustness with larger datasets.
2. Integrate Explainable AI (XAI) to improve model transparency.
3. Deploy the app as a cross-platform solution (Android/iOS).
4. Collaborate with healthcare institutions for real-world testing.

---

## 🛠️ **Technologies Used**

### **Backend**
- **Framework**: Flask/Django
- **Language**: Python
- **Database**: SQLite/PostgreSQL

### **Frontend**
- **Framework**: React.js/Flutter
- **Design**: CSS, JavaScript

### **Machine Learning**
- **Framework**: TensorFlow/Keras
- **Algorithm**: Convolutional Neural Network (CNN)

---

## 🤝 **Acknowledgments**

Special thanks to the medical and AI communities for their valuable datasets and research.  
Inspirational guidance from **Dr. Victor Ikechukwu**. Explore their work: [Dr. Victor Ikechukwu](https://github.com/Victor-Ikechukwu).

---

## ✨ **Contributors**

- **Prajwal Koundinya** - AI/ML Engineer  
  [GitHub - Prajwal Koundinya](https://github.com/Prajwal-koundinya)  
- **Team Curatherm** - Designers and Collaborators

---

