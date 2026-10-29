import streamlit as st
import librosa
import numpy as np
import pickle
from pathlib import Path
import tempfile
import os
import base64

# Page config
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

# Set background image
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

background_image_path = BASE_DIR / "microphone-2130806_1280.jpg"

with open(background_image_path, "rb") as f:
    encoded_bg = base64.b64encode(f.read()).decode()

# Apply custom CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_bg}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        font-family: 'Arial', sans-serif;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.55);
        z-index: -1;
    }}
    header[data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0.0);
        color: white;
    }}
    [data-testid="collapsedControl"] svg {{
        color: white;
    }}
    /* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #242424 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffebc6 !important;
}

/* Sidebar radio selection */
[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #ffebc6 !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1 {
    color: #ffdd99 !important;
}

/* Sidebar hover */
[data-testid="stSidebar"] button:hover {
    background-color: #393939 !important;
}
    .main-title {{
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
        color: #ffeecc;
    }}
    .main-title h4 {{
        font-size: 1.5rem;
        font-weight: normal;
        color: #ffdd99;
        margin-bottom: -0.2rem;
    }}
    .main-title h1 {{
        font-size: 4rem;
        font-weight: bold;
        color: #fff3cc;
    }}
    .sub-message {{
        text-align: center;
        font-size: 1.3rem;
        margin-top: 1rem;
        color: #ffeaa7;
    }}
    .upload-instruction {{
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin-top: 2rem;
        color: #fff1b8;
    }}
    .element-container label, .stFileUploader, .stAudio {{
        color: #fff8dc !important;
    }}
    section.main > div {{
        background-color: rgba(0, 0, 0, 0.4);
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }}
    .block-container {{
        padding-top: 1rem;
    }}
    .footer-text {{
        color: #ffeecc;
        text-align: center;
        margin-top: 2rem;
        font-size: 1.1rem;
    }}
    /* Upload widget */
[data-testid="stFileUploader"] {
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 10px;
}

/* Microphone widget */
[data-testid="stAudioInput"] {
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 10px;
}

/* General widget labels */
.stFileUploader label,
[data-testid="stAudioInput"] label {
    color: #ffebc6 !important;
}
    </style>
    <div class="main-title">
        <h4>Welcome to</h4>
        <h1>Speech Emotion Recognition System</h1>
    </div>
    <div class="sub-message">Hey Buddy 👋</div>
    <div class="upload-instruction">👉 Upload a .wav file or record your voice to detect emotion.</div>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Model training classes", "Why needed", "Owners"])

if section == "Model training classes":
    st.sidebar.write("Emotions recognized by the model:")
    emotions = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    for k, v in emotions.items():
        st.sidebar.write(f"**{k}** → {v}")
elif section == "Why needed":
    st.sidebar.write("""
    Emotion recognition helps improve:
    - Human-computer interaction  
    - Mental health detection   
    - Smart assistants  
    - Customer support experience 
    """)
elif section == "Owners":
    st.sidebar.write("""
    - Anushka Singh
    """)

# Load trained model and scaler
MODEL_PATH = PROJECT_DIR / "jupyter" / "mlp_emotion_model.pkl"
SCALER_PATH = PROJECT_DIR / "jupyter" / "scaler.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Feature extraction
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file_name, sr=None)
    result = np.array([])
    if chroma:
        stft = np.abs(librosa.stft(X))
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma_feature = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_feature))
    if mel:
        mel_feature = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel_feature))
    return result

# Upload section
uploaded_file = st.file_uploader("Upload a WAV file 🎵", type=["wav"])

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        st.audio(uploaded_file, format="audio/wav")

        features = extract_feature(temp_path).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)

        st.markdown(f"""
        <div style='background-color:#fff0b3;padding:15px 20px;border-radius:10px;
        color:#4d3900;font-weight:bold;font-size:1.2rem;text-align:center;'>
        🎧 Predicted Emotion: <strong>{prediction[0]}</strong>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

# Record section heading
st.markdown("<h2 style='color: #ffd95e; font-weight: bold;'>🎙️ Or Record your Voice</h2>", unsafe_allow_html=True)

# Microphone recording using browser
audio_value = st.audio_input("Record your voice 🎤")

if audio_value is not None:
    st.audio(audio_value)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as temp_file:
            temp_file.write(audio_value.getvalue())
            temp_path = temp_file.name

        features = extract_feature(temp_path).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)

        st.markdown(f"""
        <div style='background-color:#fff0b3;padding:15px 20px;
        border-radius:10px;color:#4d3900;font-weight:bold;
        font-size:1.2rem;text-align:center;'>
        🎧 Predicted Emotion: <strong>{prediction[0]}</strong>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Recording prediction failed: {e}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# Confusion matrix
st.markdown("""
<div style='color:#ffd95e;font-size:2rem;font-weight:bold;
margin-top:2rem;text-align:center;'>
📊 Model Confusion Matrix
</div>
""", unsafe_allow_html=True)

st.image(
    str(BASE_DIR / "confusion_matrix.png"),
    caption="Model Confusion Matrix",
    use_container_width=True
)
