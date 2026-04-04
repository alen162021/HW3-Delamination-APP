import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import tempfile
import scipy.signal as signal_processing  # Fixed import for CWT

# --- PAGE CONFIG ---
st.set_page_config(page_title="Delamination Detector", layout="wide", page_icon="🔊")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception:
    st.error("Error: Could not find 'model.pkl' in the repository.")
    st.stop()

# --- AUDIO LOADING ---
def load_audio(file):
    suffix = "." + file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    signal, sr = librosa.load(tmp_path, sr=22050)
    return signal, sr

# --- HIT DETECTION ---
def split_hits(signal, sr):
    signal = signal / (np.max(np.abs(signal)) + 1e-9)
    frame_length = int(0.02 * sr)
    hop_length = int(0.01 * sr)
    energy = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    indices = np.where(energy > np.mean(energy) * 1.5)[0]
    
    if len(indices) == 0:
        return [], []

    segments = np.split(indices, np.where(np.diff(indices) > 2)[0] + 1)
    hits, boundaries = [], []
    for seg in segments:
        start = seg[0] * hop_length
        end = seg[-1] * hop_length
        hit = signal[start:end]
        if len(hit) > 200:
            hits.append(hit)
            boundaries.append((start, end))
    return hits, boundaries

# --- FEATURE EXTRACTION ---
def extract_features(signal, sr):
    # Adjust n_fft for short signals to avoid warnings
    n_fft_adj = min(len(signal), 2048)
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=n_fft_adj), axis=1)
    psd_mean = np.mean(np.abs(np.fft.fft(signal))**2)
    return np.hstack([mfcc, psd_mean])

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Science Behind the Sound")
    with st.expander("A. Time Domain"):
        st.write("Healthy blocks ring longer; damaged ones decay faster.")
    with st.expander("B. Frequency Domain"):
        st.write("Damage reduces stiffness → shifts energy to lower frequencies.")
    with st.expander("C. Time-Frequency"):
        st.write("STFT and CWT track how sound energy moves across frequencies over time.")
    st.divider()
    st.caption("TIP | Smart Materials & Structures Lab | UH")

# --- MAIN UI ---
st.title("🔊 Composite Structural Health Monitor")
uploaded_files = st.file_uploader("Upload Audio (.wav, .m4a)", type=["wav", "m4a"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with st.expander(f"Analysis: {file.name}", expanded=True):
            signal, sr = load_audio(file)
            hits, boundaries = split_hits(signal, sr)

            if not hits:
                st.warning("No hits detected.")
                continue

            features = [extract_features(h, sr) for h in hits]
            preds = model.predict(features)
            confidence = np.mean(model.predict_proba(features)[:, 1]) if hasattr(model, "predict_proba") else np.mean(preds)
            is_bad = confidence > 0.5

            col_res, col_plot = st.columns([1, 1.5])
            with col_res:
                if is_bad: st.error("### ❌ DEFECT DETECTED")
                else: st.success("### ✅ HEALTHY")
                st.metric("Confidence", f"{confidence*100 if is_bad else (1-confidence)*100:.1f}%")

            with col_plot:
                fig_t, ax_t = plt.subplots(figsize=(7, 3))
                ax_t.plot(signal, color='gray', alpha=0.4)
                for s, e in boundaries:
                    ax_t.axvspan(s, e, color='red' if is_bad else 'green', alpha=0.3)
                st.pyplot(fig_t)
                plt.close(fig_t)

            st.divider()
            st.subheader("🔬 Multi-Domain Analysis (Hit #1)")
            sample_hit = hits[0]
            
            # Frequency Domain
            f_col1, f_col2 = st.columns(2)
            fft_vals = np.abs(np.fft.rfft(sample_hit))
            freqs = np.fft.rfftfreq(len(sample_hit), 1/sr)
            with f_col1:
                fig_fft, ax_fft = plt.subplots(figsize=(6, 3))
                ax_fft.plot(freqs, fft_vals, color='teal')
                ax_fft.set_xlim(0, 5000)
                ax_fft.set_title("FFT Magnitude")
                st.pyplot(fig_fft)
            with f_col2:
                psd = (fft_vals**2) / (len(sample_hit) * sr)
                fig_psd, ax_psd = plt.subplots(figsize=(6, 3))
                ax_psd.semilogy(freqs, psd, color='darkorange')
                ax_psd.set_xlim(0, 5000)
                ax_psd.set_title("PSD (Log Scale)")
                st.pyplot(fig_psd)

            # Time-Frequency Domain
            t1, t2, t3 = st.columns(3)
            with t1:
                st.caption("STFT Spectrogram")
                n_fft
