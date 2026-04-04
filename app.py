import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import tempfile
import scipy.signal as scipy_signal  # Changed this line

# --- PAGE CONFIG ---
st.set_page_config(page_title="Delamination Detector", layout="wide", page_icon="🔊")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception:
    st.error("Error: Could not find 'model.pkl'. Please ensure it is in the repository.")
    st.stop()

# --- AUDIO LOADING ---
def load_audio(file):
    suffix = "." + file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    # sr=22050 is standard for acoustic analysis
    signal, sr = librosa.load(tmp_path, sr=22050)
    return signal, sr

# --- HIT DETECTION ---
def split_hits(signal, sr):
    # Normalize amplitude
    signal = signal / (np.max(np.abs(signal)) + 1e-9)
    
    frame_length = int(0.02 * sr)
    hop_length = int(0.01 * sr)
    
    # RMS Energy calculation
    energy = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = np.mean(energy) * 1.5
    indices = np.where(energy > threshold)[0]
    
    if len(indices) == 0:
        return [], []

    # Group consecutive energy frames into "hits"
    segments = np.split(indices, np.where(np.diff(indices) > 2)[0] + 1)
    hits, boundaries = [], []
    
    for seg in segments:
        start = seg[0] * hop_length
        end = seg[-1] * hop_length
        hit = signal[start:end]
        if len(hit) > 200: # Ignore micro-clicks/noise
            hits.append(hit)
            boundaries.append((start, end))
    return hits, boundaries

# --- FEATURE EXTRACTION ---
def extract_features(signal, sr):
    # Fix: Ensure n_fft is not larger than the hit length
    n_fft_adj = min(len(signal), 2048)
    hop_adj = n_fft_adj // 4
    
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=n_fft_adj, hop_length=hop_adj), axis=1)
    psd_mean = np.mean(np.abs(np.fft.fft(signal))**2)
    return np.hstack([mfcc, psd_mean])

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Science Behind the Sound")
    with st.expander("A. Time Domain"):
        st.write("Healthy composites 'ring' longer. Delamination causes rapid energy decay.")
    with st.expander("B. Frequency Domain"):
        st.write("Defects reduce material stiffness, shifting peak energy to lower frequencies.")
    with st.expander("C. Time-Frequency"):
        st.write("STFT and CWT track how sound energy moves across frequencies over milliseconds.")
    st.divider()
    st.caption("TIP | Smart Materials & Structures Lab | University of Houston")

# --- MAIN UI ---
st.title("🔊 Composite Structural Health Monitor")
st.write("Upload percussion recordings to detect internal defects.")

uploaded_files = st.file_uploader("Upload Audio (.wav, .m4a)", type=["wav", "m4a"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with st.expander(f"Analysis: {file.name}", expanded=True):
            signal, sr = load_audio(file)
            hits, boundaries = split_hits(signal, sr)

            if not hits:
                st.warning("No percussion hits detected.")
                continue

            # Process all hits for the final decision
            features = [extract_features(h, sr) for h in hits]
            preds = model.predict(features)
            
            # Confidence logic
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[:, 1]
                confidence = np.mean(probs)
            else:
                confidence = np.mean(preds)

            is_bad = confidence > 0.5

            # --- DISPLAY RESULTS ---
            col_res, col_plot = st.columns([1, 1.5])
            with col_res:
                if is_bad:
                    st.error("### ❌ DEFECT DETECTED")
                    st.metric("Damage Confidence", f"{confidence*100:.1f}%")
                else:
                    st.success("### ✅ HEALTHY")
                    st.metric("Health Confidence", f"{(1-confidence)*100:.1f}%")
                st.info(f"Analyzed {len(hits)} hits")

            with col_plot:
                fig_time, ax_time = plt.subplots(figsize=(7, 3))
                ax_time.plot(signal, color='gray', alpha=0.4)
                for start, end in boundaries:
                    ax_time.axvspan(start, end, color='red' if is_bad else 'green', alpha=0.3)
                ax_time.set_title("Time Domain: Detected Hits")
                st.pyplot(fig_time)
                plt.close(fig_time)

            # --- MULTI-DOMAIN VISUALIZATION (Hit #1) ---
            st.divider()
            st.subheader("🔬 Representative Hit Analysis")
            sample_hit = hits[0]
            
            # B. FREQUENCY DOMAIN
            st.write("#### B. Frequency Domain (FFT & PSD)")
            cf1, cf2 = st.columns(2)
            fft_vals = np.abs(np.fft.rfft(sample_hit))
            freqs = np.fft.rfftfreq(len(sample_hit), 1/sr)
            
            with cf1:
                fig_fft, ax_fft = plt.subplots(figsize=(6, 3))
                ax_fft.plot(freqs, fft_vals, color='teal')
                ax_fft.set_xlim(0, 5000)
                ax_fft.set_title("FFT Magnitude")
                st.pyplot(fig_fft)
                plt.close(fig_fft)

            with cf2:
                psd = (fft_vals**2) / (len(sample_hit) * sr)
                fig_psd, ax_psd = plt.subplots(figsize=(6, 3))
                ax_psd.semilogy(freqs, psd, color='darkorange')
                ax_psd.set_xlim(0, 5000)
                ax_psd.set_title("PSD (Log Scale)")
                st.pyplot(fig_psd)
                plt.close(fig_psd)

            # C. TIME-FREQUENCY DOMAIN
            st.write("#### C. Time-Frequency Domain")
            t1, t2, t3 = st.columns(3)

            with t1:
                st.caption("STFT Spectrogram")
                n_fft_plot = min(len(sample_hit), 2048)
                stft = np.abs(librosa.stft(sample_hit, n_fft=n_fft_plot))
                fig_stft, ax_stft = plt.subplots(figsize=(5, 4))
                librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                                        y_axis='log', x_axis='time', sr=sr, ax=ax_stft)
                st.pyplot(fig_stft)
                plt.close(fig_stft)

            with t2:
                st.caption("CWT (Ricker Scalogram)")
                widths = np.arange(1, 31)
                # Scipy CWT for transient detection
                cwtmatr = scipy_signal.cwt(sample_hit, scipy_signal.ricker, widths)
                fig_cwt, ax_cwt = plt.subplots(figsize=(5, 4))
                ax_cwt.imshow(np.abs(cwtmatr), extent=[0, len(sample_hit)/sr, 1, 31], cmap='magma', aspect='auto')
                st.pyplot(fig_cwt)
                plt.close(fig_cwt)

            with t3:
                st.caption("MFCC (AI Features)")
                n_fft_mfcc = min(len(sample_hit), 2048)
                mfcc_vis = librosa.feature.mfcc(y=sample_hit, sr=sr, n_mfcc=13, n_fft=n_fft_mfcc)
                fig_m, ax_m = plt.subplots(figsize=(5, 4))
                librosa.display.specshow(mfcc_vis, x_axis='time', ax=ax_m)
                st.pyplot(fig_m)
                plt.close(fig_m)
