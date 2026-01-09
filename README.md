# 🎙️Speech Emotion Recognition using Machine Learning

This project focuses on building a **Speech Emotion Recognition (SER)** system that can identify human emotions from speech signals.  
The system is designed to classify emotions such as **happy, sad, angry, calm, fearful**, etc., using audio feature extraction and machine learning techniques.

---

## Project Overview
Understanding emotions from speech plays an important role in areas like:
- Customer support analysis
- Mental health monitoring
- Voice-based assistants
- Human–computer interaction

In this project, speech signals are processed, relevant audio features are extracted, and a machine learning model is trained to predict emotions accurately.

---

## Emotions Detected
- Happy  
- Sad  
- Angry  
- Calm  
- Fearful  
- Neutral  
*(based on RAVDESS dataset)*

---

## Technologies Used
- **Programming Language:** Python  
- **Libraries:**  
  - NumPy  
  - Pandas  
  - Librosa  
  - Scikit-learn  
  - Matplotlib  
- **Model:** Multi-Layer Perceptron (MLP Classifier)  
- **Tools:**  
  - Jupyter Notebook  
  - Streamlit  
  - VS Code  

---

## Dataset
- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
- Audio samples labeled with corresponding emotions.
- Dataset used only for training and testing purposes.

---

## Workflow
1. Load and preprocess audio files  
2. Extract audio features (MFCCs, Chroma, etc.)  
3. Normalize features using StandardScaler  
4. Train an MLP classifier  
5. Evaluate model using accuracy and confusion matrix  
6. Deploy prediction using a Streamlit web app  

---

## Model Evaluation
- Confusion Matrix used for performance analysis
- Model trained on labeled emotional speech data

---

## Web Application
The project includes a **Streamlit-based web app** that allows users to:
- Upload audio files
- Record voice input
- Predict emotion from speech in real time

---

## How to Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
