import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Delamination Detector", layout="wide", page_icon="🔊")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error: Could not find 'model.pkl'. Please ensure it is in the same folder as this script.")
    st.stop()

# --- PROCESSING FUNCTIONS ---
def load_audio(file):
    try:
        signal, sr = librosa.load(file, sr=22050)
    except:
        signal, sr = librosa.load(file, sr=22050, backend="audioread")
    return signal, sr

def split_hits(signal, sr):
    signal = signal / (np.max(np.abs(signal)) + 1e-9)
    frame_length, hop_length = int(0.02 * sr), int(0.01 * sr)
    energy = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    indices = np.where(energy > np.mean(energy) * 1.5)[0]
    if len(indices) == 0: return [], []
    segments = np.split(indices, np.where(np.diff(indices) > 2)[0] + 1)
    hits, boundaries = [], []
    for seg in segments:
        start, end = seg[0] * hop_length, seg[-1] * hop_length
        hit = signal[start:end]
        if len(hit) > 200:
            hits.append(hit)
            boundaries.append((start, end))
    return hits, boundaries

def extract_features(signal, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13), axis=1)
    psd_mean = np.mean(np.abs(np.fft.fft(signal))**2)
    return np.hstack([mfcc, psd_mean])

# --- SIDEBAR: SCIENCE & SPONSORSHIP ---
with st.sidebar:
    st.header("🔍 Science Behind the Sound")
    
    with st.expander("What is the Time Domain?"):
        st.write("It shows the sound's 'Heartbeat.' Healthy blocks ring longer; damaged blocks fade out fast due to internal friction.")
    
    with st.expander("What is the Frequency Graph?"):
        st.write("It shows the 'Pitch.' Delamination makes the block less stiff, which usually shifts the pitch to a lower frequency.")
        
    with st.expander("What are MFCCs?"):
        st.write("The 'Acoustic Fingerprint.' Our AI uses these to recognize the unique texture of a defect, much like voice recognition.")

    st.divider()
    st.markdown("### 🏛️ Acknowledgments")
    st.caption("""
    This project was partially sponsored by a **Teaching Innovation Program (TIP) grant**, 
    **Smart Materials and Structures Laboratory (SMSL)**, and 
    **Artificial Intelligent Laboratory for Monitoring and Inspection**, 
    University of Houston.
    """)

# --- MAIN UI ---
st.title("🔊 Composite Structural Health Monitor")
st.write("Upload percussion recordings to check for internal defects.")

uploaded_files = st.file_uploader("Upload Audio (.wav or .m4a)", type=["wav", "m4a"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with st.expander(f"Analysis: {file.name}", expanded=True):
            signal, sr = load_audio(file)
            hits, boundaries = split_hits(signal, sr)
            
            if not hits:
                st.warning("No hits detected.")
                continue
            
            preds = [model.predict([extract_features(h, sr)])[0] for h in hits]
            avg_score = np.mean(preds)
            is_bad = avg_score > 0.5
            
            # --- DISPLAY RESULTS ---
            col_res, col_plot = st.columns([1, 1.5])
            
            with col_res:
                if is_bad:
                    st.error(f"### Status: DEFECT DETECTED")
                else:
                    st.success(f"### Status: HEALTHY")
                st.metric("Model Certainty", f"{(avg_score if is_bad else 1-avg_score)*100:.1f}%")
                st.info(f"Analyzed {len(hits)} individual impacts.")

            with col_plot:
                fig, ax = plt.subplots(figsize=(7, 2.5))
                ax.plot(signal, color='gray', alpha=0.4)
                for start, end in boundaries:
                    ax.axvspan(start, end, color='red' if is_bad else 'green', alpha=0.3)
                ax.set_title("Detected Hits (Time Domain)")
                ax.set_axis_off()
                st.pyplot(fig)

            # --- DETAILED GRAPHS ---
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Acoustic Fingerprint (MFCC)")
                mfcc_vis = librosa.feature.mfcc(y=hits[0], sr=sr, n_mfcc=13)
                fig_m, ax_m = plt.subplots(figsize=(5, 2))
                librosa.display.specshow(mfcc_vis, ax=ax_m)
                st.pyplot(fig_m)
            
            with c2:
                st.caption("Energy Distribution (Frequency)")
                fft = np.abs(np.fft.fft(hits[0]))**2
                freqs = np.fft.fftfreq(len(hits[0]), 1/sr)
                fig_f, ax_f = plt.subplots(figsize=(5, 2))
                ax_f.plot(freqs[:len(freqs)//2], fft[:len(fft)//2], color='purple')
                ax_f.set_xlim(0, 4000)
                st.pyplot(fig_f)
