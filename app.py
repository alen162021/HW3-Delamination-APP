import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Delamination Detector", layout="wide", page_icon="🔊")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# --- TITLE ---
st.title("🔊 Delamination Detection App")
st.write("Upload an audio file to detect structural delamination.")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

# --- FEATURE EXTRACTION FUNCTION ---
def extract_features(y, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
    
    features = np.hstack([mfcc, spectral_centroid, zero_crossing])
    return features

# --- PROCESS FILE ---
if uploaded_file is not None:
    st.audio(uploaded_file)

    # Load audio
    y, sr = librosa.load(uploaded_file, sr=None)

    # Plot waveform
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # Extract features
    features = extract_features(y, sr)

    # Predict
    prediction = model.predict([features])

    # Display result
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠️ Delamination Detected")
    else:
        st.success("✅ No Delamination Detected")
