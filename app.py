import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import tempfile
from scipy import signal as scipy_signal

# --- PAGE CONFIG ---
st.set_page_config(page_title="Delamination Detector", layout="wide", page_icon="🔊")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception:
    st.error("Error: Could not find 'model.pkl'. Make sure it is uploaded to your repo.")
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
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13), axis=1)
    psd_mean = np.mean(np.abs(np.fft.fft(signal))**2)
    return np.hstack([mfcc, psd_mean])

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Science Behind the Sound")
    with st.expander("A. Time Domain"):
        st.write("Healthy blocks ring longer; damaged ones decay faster due to internal energy loss.")
    with st.expander("B. Frequency Domain"):
        st.write("Damage reduces stiffness → shifts energy to lower frequencies (FFT/PSD).")
    with st.expander("C. Time-Frequency Domain"):
        st.write("STFT and CWT show how frequency changes over time. MFCCs capture the 'acoustic fingerprint'.")
    st.divider()
    st.caption("Sponsored by TIP | Smart Materials & Structures Lab | University of Houston")

# --- MAIN UI ---
st.title("🔊 Composite Structural Health Monitor")
st.write("Upload percussion recordings to detect internal defects.")

uploaded_files = st.file_uploader("Upload Audio (.wav or .m4a)", type=["wav", "m4a"], accept_multiple_files=True)

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
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[:, 1]
                confidence = np.mean(probs)
            else:
                confidence = np.mean(preds)

            is_bad = confidence > 0.5

            # --- RESULTS SUMMARY ---
            col_res, col_plot = st.columns([1, 1.5])
            with col_res:
                if is_bad:
                    st.error("### ❌ DEFECT DETECTED")
                    st.metric("Confidence (Damage)", f"{confidence*100:.1f}%")
                else:
                    st.success("### ✅ HEALTHY")
                    st.metric("Confidence (Healthy)", f"{(1-confidence)*100:.1f}%")
                st.info(f"Analyzed {len(hits)} hits")

            with col_plot:
                fig_time, ax_time = plt.subplots(figsize=(7, 3))
                ax_time.plot(signal, color='gray', alpha=0.4)
                for start, end in boundaries:
                    ax_time.axvspan(start, end, color='red' if is_bad else 'green', alpha=0.3)
                ax_time.set_title("Detected Hits (Time Domain)")
                st.pyplot(fig_time)

            # --- DETAILED ANALYSIS GRAPHS ---
            st.divider()
            st.subheader("🔬 Multi-Domain Signal Analysis (Representative Hit)")
            sample_hit = hits[0]
            
            # B. FREQUENCY DOMAIN
            st.write("#### B. Frequency Domain (FFT & PSD)")
            col_f1, col_f2 = st.columns(2)
            fft_vals = np.abs(np.fft.rfft(sample_hit))
            freqs = np.fft.rfftfreq(len(sample_hit), 1/sr)
            
            with col_f1:
                fig_fft, ax_fft = plt.subplots(figsize=(6, 3))
                ax_fft.plot(freqs, fft_vals, color='teal')
                ax_fft.set_title("FFT (Linear Magnitude)")
                ax_fft.set_xlabel("Frequency (Hz)")
                ax_fft.set_xlim(0, 5000)
                st.pyplot(fig_fft)

            with col_f2:
                psd = (fft_vals**2) / (len(sample_hit) * sr)
                fig_psd, ax_psd = plt.subplots(figsize=(6, 3))
                ax_psd.semilogy(freqs, psd, color='darkorange')
                ax_psd.set_title("PSD (Power Spectral Density)")
                ax_psd.set_xlabel("Frequency (Hz)")
                ax_psd.set_xlim(0, 5000)
                st.pyplot(fig_psd)

            # C. TIME-FREQUENCY DOMAIN
            st.write("#### C. Time-Frequency Domain (STFT, CWT, MFCC)")
            t1, t2, t3 = st.columns(3)

            with t1:
                st.caption("STFT (Spectrogram)")
                stft = np.abs(librosa.stft(sample_hit))
                fig_stft, ax_stft = plt.subplots(figsize=(5, 4))
                librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                                        y_axis='log', x_axis='time', sr=sr, ax=ax_stft)
                st.pyplot(fig_stft)

            with t2:
                st.caption("CWT (Scalogram)")
                widths = np.arange(1, 31)
                # Robust Ricker call to prevent AttributeError
                cwtmatr = scipy_signal.cwt(sample_hit, scipy_signal.ricker, widths)
                fig_cwt, ax_cwt = plt.subplots(figsize=(5, 4))
                ax_cwt.imshow(np.abs(cwtmatr), extent=[0, len(sample_hit)/sr, 1, 31], 
                              cmap='magma', aspect='auto')
                ax_cwt.set_ylabel("Scale")
                st.pyplot(fig_cwt)

            with t3:
                st.caption("MFCC (Acoustic Fingerprint)")
                mfcc_vis = librosa.feature.mfcc(y=sample_hit, sr=sr, n_mfcc=13)
                fig_m, ax_m = plt.subplots(figsize=(5, 4))
                librosa.display.specshow(mfcc_vis, x_axis='time', ax=ax_m)
                st.pyplot(fig_m)
