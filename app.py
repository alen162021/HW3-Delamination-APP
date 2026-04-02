import streamlit as st
import numpy as np
import librosa
import joblib
import os

st.title("🔊 Delamination Detection App")

# =========================
# Load model
# =========================
model = joblib.load("model.pkl")

# =========================
# Audio loader (FIXED for m4a)
# =========================
def load_audio(file):
    try:
        signal, sr = librosa.load(file, sr=22050)
    except:
        st.warning("Standard load failed, retrying with audioread...")
        signal, sr = librosa.load(file, sr=22050, backend="audioread")
    return signal, sr

# =========================
# HIT SPLITTING (IMPORTANT)
# =========================
def split_hits(signal, sr):
    signal = signal / np.max(np.abs(signal))

    frame_length = int(0.02 * sr)
    hop_length = int(0.01 * sr)

    energy = librosa.feature.rms(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    threshold = np.mean(energy) * 1.5
    active = energy > threshold

    indices = np.where(active)[0]

    if len(indices) == 0:
        return []

    segments = np.split(indices, np.where(np.diff(indices) > 2)[0] + 1)

    hits = []
    for seg in segments:
        start = seg[0] * hop_length
        end = seg[-1] * hop_length
        hit = signal[start:end]

        if len(hit) > 200:
            hits.append(hit)

    return hits

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    fft = np.abs(np.fft.fft(signal))**2
    psd = fft[:len(fft)//2]
    psd_mean = np.mean(psd)

    return np.hstack([mfcc_mean, psd_mean])

# =========================
# FILE PROCESSING
# =========================
def process_file(file, folder_name="Unknown"):
    signal, sr = load_audio(file)

    hits = split_hits(signal, sr)

    if len(hits) == 0:
        return "No hits detected", []

    predictions = []

    for hit in hits:
        features = extract_features(hit, sr)
        pred = model.predict([features])[0]
        predictions.append(pred)

    # Majority vote
    final = int(round(np.mean(predictions)))

    return final, predictions

# =========================
# UI: Upload multiple files
# =========================
uploaded_files = st.file_uploader(
    "Upload audio files or entire dataset",
    type=["wav", "m4a"],
    accept_multiple_files=True
)

if uploaded_files:

    st.subheader("📊 Results")

    for file in uploaded_files:

        # Extract folder name if present
        if "/" in file.name:
            folder = file.name.split("/")[0]
        else:
            folder = "Single File"

        result, preds = process_file(file, folder)

        st.write(f"📁 **Folder:** {folder}")
        st.write(f"📄 **File:** {file.name}")

        if result == "No hits detected":
            st.warning("No hits detected")
            continue

        if result == 0:
            st.success("Prediction: GOOD")
        else:
            st.error("Prediction: BAD")

        st.write(f"Hit Predictions: {preds}")
        st.write("---")
