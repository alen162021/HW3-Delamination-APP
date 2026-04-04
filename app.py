import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import tempfile
import scipy.signal as scipy_signal
from scipy.signal import cwt, ricker  # Explicit functions for CWT

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Ensure this filename matches exactly (case-sensitive)
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model Load Error: {e}")
    st.stop()

# --- AUDIO PROCESSING ---
def load_audio(file):
    suffix = "." + file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    signal, sr = librosa.load(tmp_path, sr=22050)
    return signal, sr

# --- MAIN UI ---
st.title("🔊 Composite Structural Health Monitor")
uploaded_file = st.file_uploader("Upload Percussion Audio", type=["wav", "m4a"])

if uploaded_file:
    signal, sr = load_audio(uploaded_file)
    
    # Feature Extraction logic here...
    # (Assuming you have your hit detection/feature code)
    
    st.subheader("🔬 Time-Frequency Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("Short-Time Fourier Transform (STFT)")
        # Dynamically adjust n_fft to avoid the "n_fft too large" warning
        n_fft_adj = min(len(signal), 2048)
        stft = np.abs(librosa.stft(signal, n_fft=n_fft_adj))
        fig, ax = plt.subplots()
        librosa.display.specshow(librosa.amplitude_to_db(stft), y_axis='log', x_axis='time', sr=sr)
        st.pyplot(fig)

    with col2:
        st.caption("Continuous Wavelet Transform (CWT)")
        widths = np.arange(1, 31)
        # Using the direct function calls to prevent AttributeError
        cwtmatr = cwt(signal, ricker, widths)
        fig_c, ax_c = plt.subplots()
        ax_c.imshow(np.abs(cwtmatr), extent=[0, len(signal)/sr, 1, 31], cmap='magma', aspect='auto')
        st.pyplot(fig_c)
