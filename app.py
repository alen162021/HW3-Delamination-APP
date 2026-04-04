import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Delamination Detector", layout="wide")

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Ensure model.pkl is in the same directory
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# --- PROCESSING FUNCTIONS ---
def load_audio(file):
    try:
        # Load with a fixed sample rate for consistency with your training
        signal, sr = librosa.load(file, sr=22050)
        return signal, sr
    except Exception:
        # Fallback for complex formats like m4a
        signal, sr = librosa.load(file, sr=22050, backend="audioread")
        return signal, sr

def split_hits(signal, sr):
    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-9)
    
    frame_length = int(0.02 * sr)
    hop_length = int(0.01 * sr)
    
    energy = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = np.mean(energy) * 1.5
    indices = np.where(energy > threshold)[0]
    
    if len(indices) == 0: return [], []

    segments = np.split(indices, np.where(np.diff(indices) > 2)[0] + 1)
    hits = []
    hit_boundaries = []
    
    for seg in segments:
        start = seg[0] * hop_length
        end = seg[-1] * hop_length
        hit = signal[start:end]
        if len(hit) > 200:
            hits.append(hit)
            hit_boundaries.append((start, end))
    return hits, hit_boundaries

def extract_features(signal, sr):
    # Match the features from your MLHW3 Colab
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    fft = np.abs(np.fft.fft(signal))**2
    psd_mean = np.mean(fft[:len(fft)//2])
    return np.hstack([mfcc_mean, psd_mean])

# --- UI LAYOUT ---
st.title("🔊 Composite Structural Health Monitor")
st.write("Upload percussion recordings to detect internal delamination using Machine Learning.")

with st.sidebar:
    st.header("Help & Info")
    st.markdown("""
    **What is Delamination?**
    It's an internal separation of composite layers. 
    **The Physics:**
    - **Good Blocks:** High stiffness, long ring, high frequency.
    - **Bad Blocks:** Low stiffness, dull thud, rapid damping.
    """)
    

uploaded_files = st.file_uploader("Choose Audio Files", type=["wav", "m4a"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with st.expander(f"Analysis: {file.name}", expanded=True):
            signal, sr = load_audio(file)
            hits, boundaries = split_hits(signal, sr)
            
            if not hits:
                st.warning("No clear impact hits detected. Check audio volume.")
                continue
            
            # Run Predictions
            raw_preds = [model.predict([extract_features(h, sr)])[0] for h in hits]
            avg_val = np.mean(raw_preds)
            is_bad = avg_val > 0.5
            confidence = avg_val if is_bad else (1 - avg_val)
            
            # --- RESULTS COLUMNS ---
            col_res, col_plot = st.columns([1, 2])
            
            with col_res:
                if is_bad:
                    st.error("### Status: DEFECT (Bad)")
                else:
                    st.success("### Status: HEALTHY (Good)")
                
                st.metric("Model Confidence", f"{confidence*100:.1f}%")
                st.write(f"Detected {len(hits)} individual hits.")

            with col_plot:
                # Waveform Visualization
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(signal, color='#bdc3c7', alpha=0.7)
                for start, end in boundaries:
                    ax.axvspan(start, end, color='#e74c3c' if is_bad else '#2ecc71', alpha=0.4)
                ax.set_title("Time-Domain Impact Detection")
                ax.set_axis_off()
                st.pyplot(fig)
                

            # --- WHY IS IT BAD? (Deep Dive) ---
            st.divider()
            st.subheader("🔍 Scientific Breakdown")
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Acoustic Fingerprint (MFCCs)**")
                # Show heatmap of the first hit
                mfccs = librosa.feature.mfcc(y=hits[0], sr=sr, n_mfcc=13)
                fig_m, ax_m = plt.subplots()
                librosa.display.specshow(mfccs, x_axis='time', ax=ax_m)
                st.pyplot(fig_m)
                
            
            with c2:
                st.write("**Frequency Energy (PSD)**")
                # Show Power Spectral Density
                fft_vals = np.abs(np.fft.fft(hits[0]))**2
                freqs = np.fft.fftfreq(len(hits[0]), 1/sr)
                fig_f, ax_f = plt.subplots()
                ax_f.plot(freqs[:len(freqs)//2], fft_vals[:len(fft_vals)//2], color='purple')
                ax_f.set_xlim(0, 5000) # Focusing on typical impact range
                ax_f.set_xlabel("Frequency (Hz)")
                st.pyplot(fig_f)
