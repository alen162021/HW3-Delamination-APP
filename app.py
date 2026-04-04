import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import tempfile
import scipy.signal as signal_processing
from scipy.signal import cwt, ricker  # Explicit import to fix AttributeError

# --- PAGE CONFIG ---
st.set_page_config(page_title="UH Composite Health Monitor", layout="wide", page_icon="🔊")

# --- LOAD TRAINED MODEL ---
@st.cache_resource
def load_model():
    # Ensure 'model.pkl' is in your GitHub root directory
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model.pkl: {e}")
    st.stop()

# --- AUDIO UTILITIES ---
def load_audio(file):
    suffix = "." + file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    # 22050Hz is standard for librosa and matches most training sets
    signal, sr = librosa.load(tmp_path, sr=22050)
    return signal, sr

def extract_features(signal, sr):
    # DYNAMIC N_FFT: Fixes the 'n_fft too large' warning in your logs
    n_fft_adj = min(len(signal), 2048)
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=n_fft_adj), axis=1)
    psd = np.mean(np.abs(np.fft.fft(signal, n=n_fft_adj))**2)
    return np.hstack([mfcc, psd])

# --- MAIN INTERFACE ---
st.title("🔊 Composite Structural Health Monitor")
st.caption("Smart Materials & Structures Lab | University of Houston")

uploaded_file = st.file_uploader("Upload Percussion Audio (WAV/M4A)", type=["wav", "m4a"])

if uploaded_file:
    signal, sr = load_audio(uploaded_file)
    
    # Run Prediction
    features = extract_features(signal, sr).reshape(1, -1)
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features).max() if hasattr(model, "predict_proba") else 1.0

    # Display Results
    res_col, plot_col = st.columns([1, 2])
    
    with res_col:
        if prediction == 1: # Assuming 1 is 'Defect'
            st.error("### ❌ DEFECT DETECTED")
            st.write("The signal shows characteristic damping associated with delamination.")
        else:
            st.success("### ✅ HEALTHY")
            st.write("The structural response indicates high stiffness and integrity.")
        st.metric("Confidence Score", f"{confidence*100:.1f}%")

    with plot_col:
        st.subheader("Time Domain (Raw Impact)")
        fig_t, ax_t = plt.subplots(figsize=(8, 3))
        ax_t.plot(signal, color='#2ecc71' if prediction == 0 else '#e74c3c')
        ax_t.set_xlabel("Samples")
        ax_t.set_ylabel("Amplitude")
        st.pyplot(fig_t)

    st.divider()
    
    # Advanced Signal Analysis
    st.subheader("🔬 Multi-Domain Diagnostic Analysis")
    t1, t2 = st.columns(2)
    
    with t1:
        st.caption("Short-Time Fourier Transform (STFT)")
        n_fft_adj = min(len(signal), 2048)
        stft = np.abs(librosa.stft(signal, n_fft=n_fft_adj))
        fig_s, ax_s = plt.subplots()
        librosa.display.specshow(librosa.amplitude_to_db(stft), y_axis='log', x_axis='time', sr=sr, ax=ax_s)
        st.pyplot(fig_s)
