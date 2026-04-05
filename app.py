import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import tempfile
import scipy.signal as signal_lib

# --- PAGE CONFIG ---
st.set_page_config(page_title="Delamination Detector Pro", layout="wide", page_icon="🔊")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception:
    st.error("Error: Could not find 'model.pkl'.")
    st.stop()

# --- AUDIO LOADING ---
def load_audio(file):
    suffix = "." + file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    sig, sr = librosa.load(tmp_path, sr=22050)
    return sig, sr

# --- HIT DETECTION ---
def split_hits(sig, sr):
    sig_norm = sig / (np.max(np.abs(sig)) + 1e-9)
    energy = librosa.feature.rms(y=sig_norm, frame_length=441, hop_length=220)[0]
    indices = np.where(energy > np.mean(energy) * 1.5)[0]
    if len(indices) == 0: return [], []
    segments = np.split(indices, np.where(np.diff(indices) > 2)[0] + 1)
    hits, boundaries = [], []
    for seg in segments:
        start, end = seg[0] * 220, seg[-1] * 220
        hit = sig_norm[start:end]
        if len(hit) > 200:
            hits.append(hit); boundaries.append((start, end))
    return hits, boundaries

# --- UI HEADER ---
st.title("🔊 Composite Structural Health Monitor: Advanced Analysis")
st.markdown("---")

uploaded_files = st.file_uploader("Upload Audio (.wav or .m4a)", type=["wav", "m4a"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with st.expander(f"Detailed Analysis: {file.name}", expanded=True):
            sig, sr = load_audio(file)
            hits, boundaries = split_hits(sig, sr)

            if not hits:
                st.warning("No hits detected.")
                continue

            # Prediction Logic
            features = [np.hstack([np.mean(librosa.feature.mfcc(y=h, sr=sr, n_mfcc=13), axis=1), 
                                  np.mean(np.abs(np.fft.fft(h))**2)]) for h in hits]
            preds = model.predict(features)
            is_bad = np.mean(preds) > 0.5
            
            # --- 2. PRESENT DATA ANALYSIS RESULTS ---
            st.header("2. Data Analysis Results")
            
            # A. Time Domain
            st.subheader("A. Time Domain (Time Series)")
            fig_t, ax_t = plt.subplots(figsize=(10, 3))
            ax_t.plot(sig, color='gray', alpha=0.5, label="Raw Signal")
            for start, end in boundaries:
                ax_t.axvspan(start, end, color='red' if is_bad else 'green', alpha=0.3)
            st.pyplot(fig_t)

            # B. Frequency Domain
            st.subheader("B. Frequency Domain (FFT & PSD)")
            col_f1, col_f2 = st.columns(2)
            sample_hit = hits[0]
            fft_res = np.abs(np.fft.fft(sample_hit))
            freqs = np.fft.fftfreq(len(sample_hit), 1/sr)
            
            with col_f1:
                fig_fft, ax_fft = plt.subplots()
                ax_fft.plot(freqs[:len(freqs)//2], fft_res[:len(fft_res)//2])
                ax_fft.set_title("Fast Fourier Transform (FFT)")
                st.pyplot(fig_fft)
            with col_f2:
                fig_psd, ax_psd = plt.subplots()
                f, Pxx_den = signal_lib.welch(sample_hit, sr)
                ax_psd.semilogy(f, Pxx_den)
                ax_psd.set_title("Power Spectral Density (PSD)")
                st.pyplot(fig_psd)

            # C. Time-Frequency Domain
            st.subheader("C. Time-Frequency Domain")
            tabs = st.tabs(["STFT", "CWT (Approximation)", "MFCC"])
            
            with tabs[0]:
                fig_stft, ax_stft = plt.subplots()
                D = librosa.amplitude_to_db(np.abs(librosa.stft(sample_hit)), ref=np.max)
                librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax_stft)
                st.pyplot(fig_stft)
            
            with tabs[1]:
                # Using a Scalogram approach via librosa CQT as a proxy for CWT
                fig_cwt, ax_cwt = plt.subplots()
                C = librosa.amplitude_to_db(np.abs(librosa.cqt(sample_hit, sr=sr)), ref=np.max)
                librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax_cwt)
                ax_cwt.set_title("Constant-Q Transform (CWT-like)")
                st.pyplot(fig_cwt)
                
            with tabs[2]:
                fig_mfcc, ax_mfcc = plt.subplots()
                mfccs = librosa.feature.mfcc(y=sample_hit, sr=sr, n_mfcc=13)
                librosa.display.specshow(mfccs, x_axis='time', ax=ax_mfcc)
                ax_mfcc.set_title("MFCC Coefficients")
                st.pyplot(fig_mfcc)

            # --- 3. OBSERVATION AND DISCUSSION ---
            st.markdown("---")
            st.header("3. Observation and Discussion")
            
            obs_col1, obs_col2 = st.columns(2)
            
            with obs_col1:
                st.subheader("A. Comparison of Methods")
                st.write("""
                * **Time Domain:** Best for identifying transient hits and decay rates, but lacks depth regarding material stiffness.
                * **Frequency Domain (FFT/PSD):** Excellent for identifying resonance shifts. PSD provides a cleaner, averaged power estimate than raw FFT.
                * **Time-Frequency:** Essential for non-stationary signals. **STFT** provides linear resolution, while **CWT** is better for capturing high-frequency transients. **MFCC** compresses this data into a 'fingerprint' ideal for ML.
                """)
            
            with obs_col2:
                st.subheader("B. Healthy vs. Unhealthy Analysis")
                if is_bad:
                    st.warning("**Current Sample: Unhealthy (Defect Detected)**")
                    st.write("""
                    - **Time:** Noticeable attenuation or irregular amplitude spikes.
                    - **Frequency:** Shift in peak frequencies toward the lower end (loss of stiffness).
                    - **Time-Frequency:** High-frequency energy dissipates more rapidly compared to healthy signatures.
                    """)
                else:
                    st.success("**Current Sample: Healthy**")
                    st.write("""
                    - **Time:** Consistent, clean logarithmic decay.
                    - **Frequency:** High-energy peaks at expected fundamental frequencies.
                    - **Time-Frequency:** Stable harmonics across the duration of the hit.
                    """)

            st.caption("**Note:** These results are based on a limited dataset and may not be representative of all material types or defect geometries.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings & Info")
    st.info("This tool uses multi-domain analysis to detect delamination in composite structures.")
    st.divider()
    st.caption("Developed for University of Houston | Smart Materials Lab")
