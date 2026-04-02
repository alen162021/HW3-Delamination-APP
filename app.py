

import streamlit as st
import numpy as np
import librosa
import joblib

import os
st.write("Files in directory:", os.listdir())

st.title("Delamination Detection using Acoustic Signals")

# Load trained model
model = joblib.load("model.pkl")

# Feature extraction (same as your training)
def extract_features(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    fft = np.abs(np.fft.fft(signal))**2
    psd = fft[:len(fft)//2]
    psd_mean = np.mean(psd)

    return np.hstack([mfcc_mean, psd_mean])

# Upload audio
uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    signal, sr = librosa.load(uploaded_file, sr=22050)
    features = extract_features(signal, sr)

    prediction = model.predict([features])

    if prediction[0] == 0:
        st.success("Prediction: GOOD (No Delamination)")
    else:
        st.error("Prediction: BAD (Delamination Detected)")
