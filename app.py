import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Delamination Detector", page_icon="🔊")

st.title("🔊 Composite Delamination Detection")
st.markdown("""
This app uses **Machine Learning** to analyze the acoustic signature of percussion hits on composite blocks. 
It identifies internal delamination (defects) that are often invisible to the naked eye.
""")

# =========================
# Load model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# =========================
# Processing Functions
# =========================
def load_audio(file):
    try:
        signal, sr = librosa.load(file, sr=22050)
    except:
        signal, sr = librosa.load(file, sr=22050, backend="audioread")
    return signal, sr

def split_hits(signal, sr):
    signal = signal / (np.max(np.abs(signal)) + 1e-9)
    frame_length = int(0.02 * sr)
    hop_length = int(0.01 * sr)
    energy = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    
    threshold = np.mean(energy) * 1.5
    indices = np.where(energy > threshold)[0]
    
    if len(indices) == 0: return []
    
    segments = np.split(indices, np.where(np.diff(indices) > 2)[0] + 1)
    hits = []
    hit_times = [] # To store for plotting
    for seg in segments:
        start = seg[0] * hop_length
        end = seg[-1] * hop_length
        hit = signal[start:end]
        if len(hit) > 200:
            hits.append(hit)
            hit_times.append((start, end))
    return hits, hit_times

def extract_features(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    fft = np.abs(np.fft.fft(signal))**2
    psd_mean = np.mean(fft[:len(fft)//2])
    return np.hstack([mfcc_mean, psd_mean])

# =========================
# UI Sidebar - Educational Info
# =========================
with st.sidebar:
    st.header("How it Works")
    st.info("""
    **1. Hit Detection:** The app finds 'spikes' in audio energy.
    **2. Feature Extraction:** It calculates MFCCs (acoustic fingerprints) and PSD (energy distribution).
    **3. Classification:** A pre-trained model compares these features to known 'Good' and 'Bad' samples.
    """)
    

# =========================
# Main Upload Logic
# =========================
uploaded_files = st.file_uploader(
    "Upload audio files (.wav or .m4a)", 
    type=["wav", "m4a"], 
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        with st.expander(f"Analysis for: {file.name}", expanded=True):
            signal, sr = load_audio(file)
            hits, hit_times = split_hits(signal, sr)
            
            if not hits:
                st.warning("No percussion hits detected in this file.")
                continue
            
            # Prediction Logic
            preds = [model.predict([extract_features(h, sr)])[0] for h in hits]
            avg_score = np.mean(preds)
            is_bad = avg_score > 0.5
            confidence = avg_score if is_bad else (1 - avg_score)
            
            # Display Result
            col1, col2 = st.columns(2)
            with col1:
                if is_bad:
                    st.error(f"### Result: BAD (Defect)")
                else:
                    st.success(f"### Result: GOOD (Healthy)")
                st.metric("Confidence", f"{confidence*100:.1f}%")

            with col2:
                # Plot the hits found
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.plot(signal, color='gray', alpha=0.5)
                for start, end in hit_times:
                    ax.axvspan(start, end, color='red' if is_bad else 'green', alpha=0.3)
                ax.set_title("Detected Hits in Recording")
                ax.set_axis_off()
                st.pyplot(fig)
