import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import tempfile

# --- PAGE CONFIG ---
st.set_page_config(page_title="Delamination Detector", layout="wide", page_icon="🔊")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception:
    st.error("Error: Could not find 'model.pkl'. Ensure it is in your GitHub repo.")
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
    n_fft_val = min(len(signal), 2048)
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=n_fft_val), axis=1)
    psd_mean = np.mean(np.abs(np.fft.fft(signal))**2)
    return np.hstack([mfcc, psd_mean])

# --- MAIN UI ---
st.title("🔊 Composite Structural Health Monitor")
uploaded_files = st.file_uploader("Upload Audio", type=["wav", "m4a"], accept_multiple_files=True)

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

            res_col, plot_col = st.columns([1, 1.5])
            with res_col:
                if is_bad: st.error("### ❌ DEFECT DETECTED")
                else: st.success("### ✅ HEALTHY")
                st.metric("Confidence", f"{confidence*100 if is_bad else (1-confidence)*100:.1f}%")

            with plot_col:
                fig_t, ax_t = plt.subplots(figsize=(7, 2.5))
                ax_t.plot(signal, color='gray', alpha=0.4)
                for s, e in boundaries:
                    ax_t.axvspan(s, e, color='red' if is_bad else 'green', alpha=0.3)
                st.pyplot(fig_t)
                plt.close(fig_t)

            st.divider()
            st.subheader("🔬 Multi-Domain Analysis (Hit #1)")
            sample_hit = hits[0]
            
            # Frequency Domain
            f1, f2 = st.columns(2)
            fft_vals = np.abs(np.fft.rfft(sample_hit))
            freqs = np.fft.rfftfreq(len(sample_hit), 1/sr)
            with f1:
                fig_f, ax_f = plt.subplots(figsize=(6, 3))
                ax_f.plot(freqs, fft_vals, color='teal')
                ax_f.set_xlim(0, 5000)
                ax_f.set_title("FFT Magnitude")
                st.pyplot(fig_f)
            with f2:
                psd = (fft_vals**2) / (len(sample_hit) * sr)
                fig_p, ax_p = plt.subplots(figsize=(6, 3))
                ax_p.semilogy(freqs, psd, color='darkorange')
                ax_p.set_xlim(0, 5000)
                ax_p.set_title("PSD (Power Spectral Density)")
                st.pyplot(fig_p)

            # Time-Frequency Domain
            t1, t2, t3 = st.columns(3)
            n_fft_plot = min(len(sample_hit), 2048)

            with t1:
                st.caption("STFT Spectrogram")
                stft = np.abs(librosa.stft(sample_hit, n_fft=n_fft_plot))
                fig_s, ax_s = plt.subplots(figsize=(5, 4))
                librosa.display.specshow(librosa.amplitude_to_db(stft), y_axis='log', sr=sr, ax=ax_s)
                st.pyplot(fig_s)
            
            with t2:
                st.caption("CWT (Ricker Scalogram)")
                try:
                    # Lazy import to ensure the rest of the app loads if SciPy is slow
                    from scipy.signal import cwt, ricker
                    widths = np.arange(1, 31)
                    cwtmatr = cwt(sample_hit, ricker, widths)
                    fig_c, ax_c = plt.subplots(figsize=(5, 4))
                    ax_c.imshow(np.abs(cwtmatr), extent=[0, len(sample_hit)/sr, 1, 31], cmap='magma', aspect='auto')
                    st.pyplot(fig_c)
                except ImportError:
                    st.error("CWT Error: SciPy 'signal' module not found. Check requirements.txt.")

            with t3:
                st.caption("MFCC Fingerprint")
                mfcc_v = librosa.feature.mfcc(y=sample_hit, sr=sr, n_mfcc=13, n_fft=n_fft_plot)
                fig_m, ax_m = plt.subplots(figsize=(5, 4))
                librosa.display.specshow(mfcc_v, ax=ax_m)
                st.pyplot(fig_m)
            
            plt.close('all')
