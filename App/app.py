import streamlit as st
import librosa
import numpy as np
import pickle
import os
import base64

# ---------------- ENV DETECTION ----------------
try:
    import sounddevice as sd
    from scipy.io.wavfile import write
    MIC_AVAILABLE = True
except ModuleNotFoundError:
    MIC_AVAILABLE = False

st.set_page_config(
    page_title="Speech Emotion Recognition",
    layout="centered"
)

# ---------------- BACKGROUND IMAGE ----------------
bg_path = "App/microphone-2130806_1280.jpg"

with open(bg_path, "rb") as f:
    encoded_bg = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/webp;base64,{encoded_bg}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
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
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- TITLE ----------------
st.markdown("""
<div style="text-align:center; color:#fff;">
    <h1>Speech Emotion Recognition System</h1>
    <p>Upload a WAV file or record your voice to detect emotion</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Model training classes", "Why needed", "Owners"])

if section == "Model training classes":
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
    - Human-computer interaction  
    - Mental health monitoring  
    - Smart assistants  
    - Customer support systems
    """)

elif section == "Owners":
    st.sidebar.write("""
    - Anushka Singh  
    - Lovely Sharma  
    - Ayush Sharma
    """)

# ---------------- LOAD MODEL ----------------
with open("App/mlp_emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("App/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------- FEATURE EXTRACTION ----------------
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name, sr=None)

    stft = np.abs(librosa.stft(X))

    mfccs = np.mean(
        librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,
        axis=0
    )

    chroma = np.mean(
        librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
        axis=0
    )

    mel = np.mean(
        librosa.feature.melspectrogram(y=X, sr=sample_rate).T,
        axis=0
    )

    return np.hstack((mfccs, chroma, mel))

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload WAV file 🎵", type=["wav"])
audio_path = None

if uploaded_file:
    with open("uploaded.wav", "wb") as f:
        f.write(uploaded_file.read())

    audio_path = "uploaded.wav"
    st.audio(audio_path)

# ---------------- MICROPHONE SECTION ----------------
st.markdown("## 🎙️ Record Your Voice")

if not MIC_AVAILABLE:
    st.info("🎤 Microphone recording is not supported in cloud environment. Please upload a WAV file.")
else:
    if st.button("Record with Microphone"):
        fs = 44100
        duration = 5

        recording = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=1
        )
        sd.wait()

        write("mic_recording.wav", fs, recording)
        audio_path = "mic_recording.wav"
        st.audio(audio_path)

# ---------------- PREDICTION ----------------
if audio_path:
    features = extract_feature(audio_path).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)

    st.success(f"🎧 Predicted Emotion: **{prediction[0]}**")
    st.image(
        "App/confusion_matrix.png",
        caption="Model Confusion Matrix",
        use_container_width=True
    )
