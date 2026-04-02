import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import os
from moviepy.editor import VideoFileClip
import pandas as pd

st.title("Delamination Detection (Audio + Video Support)")

# Load trained model
model = joblib.load("model.pkl")

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    fft = np.abs(np.fft.fft(signal))**2
    psd = fft[:len(fft)//2]
    psd_mean = np.mean(psd)

    return np.hstack([mfcc_mean, psd_mean])

# ---------------- LOAD AUDIO ----------------
def load_audio_file(file):
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # If MP4 → extract audio
        if file.name.endswith(".mp4"):
            video = VideoFileClip(tmp_path)
            audio_path = tmp_path + ".wav"
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            signal, sr = librosa.load(audio_path, sr=22050)

        else:  # WAV
            signal, sr = librosa.load(tmp_path, sr=22050)

        return signal, sr

    except Exception as e:
        st.error(f"Error processing {file.name}: {e}")
        return None, None

# ---------------- UI ----------------
uploaded_files = st.file_uploader(
    "Upload WAV or MP4 files (single or multiple)",
    type=["wav", "mp4"],
    accept_multiple_files=True
)

# ---------------- PROCESS FILES ----------------
if uploaded_files:
    results = []

    for file in uploaded_files:
        st.write(f"Processing: {file.name}")

        signal, sr = load_audio_file(file)

        if signal is None:
            continue

        features = extract_features(signal, sr)
        prediction = model.predict([features])[0]

        label = "GOOD" if prediction == 0 else "BAD"

        results.append({
            "File Name": file.name,
            "Prediction": label
        })

        st.success(f"{file.name} → {label}")

    # Show results table
    df = pd.DataFrame(results)
    st.subheader("Summary Results")
    st.dataframe(df)
