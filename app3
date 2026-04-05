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
    st.error("Error: Could not find 'model.pkl'. Make sure it is uploaded to your repo.")
    st.stop()

# --- AUDIO LOADING (FIXED FOR M4A/WAV) ---
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

    energy = librosa.feature.rms(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

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

    with st.expander("What is the Time Domain?"):
        st.write("Healthy blocks ring longer; damaged ones decay faster due to internal energy loss.")

    with st.expander("What is the Frequency Graph?"):
        st.write("Damage reduces stiffness → shifts energy to lower frequencies.")

    with st.expander("What are MFCCs?"):
        st.write("MFCCs capture the acoustic fingerprint used by the AI for classification.")

    st.divider()
    st.caption("""
    Sponsored by Teaching Innovation Program (TIP),  
    Smart Materials and Structures Lab,  
    University of Houston
    """)

# --- MAIN UI ---
st.title("🔊 Composite Structural Health Monitor")
st.write("Upload percussion recordings to detect internal defects.")

uploaded_files = st.file_uploader(
    "Upload Audio (.wav or .m4a)",
    type=["wav", "m4a"],
    accept_multiple_files=True
)

# --- PROCESS FILES ---
if uploaded_files:
    for file in uploaded_files:

        with st.expander(f"Analysis: {file.name}", expanded=True):

            signal, sr = load_audio(file)
            hits, boundaries = split_hits(signal, sr)

            if not hits:
                st.warning("No hits detected.")
                continue

            # --- FEATURE EXTRACTION ---
            features = [extract_features(h, sr) for h in hits]

            preds = model.predict(features)

            # --- CONFIDENCE (FIXED) ---
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[:, 1]
                confidence = np.mean(probs)
            else:
                confidence = np.mean(preds)

            is_bad = confidence > 0.5

            # --- RESULTS DISPLAY ---
            col_res, col_plot = st.columns([1, 1.5])

            with col_res:
                if is_bad:
                    st.error("### ❌ DEFECT DETECTED")
                    st.metric("Confidence (Damage)", f"{confidence*100:.1f}%")
                else:
                    st.success("### ✅ HEALTHY")
                    st.metric("Confidence (Healthy)", f"{(1-confidence)*100:.1f}%")

                st.info(f"Analyzed {len(hits)} hits")

            # --- TIME DOMAIN PLOT ---
            with col_plot:
                fig, ax = plt.subplots(figsize=(7, 2.5))
                ax.plot(signal, color='gray', alpha=0.4)

                for start, end in boundaries:
                    ax.axvspan(start, end,
                               color='red' if is_bad else 'green',
                               alpha=0.3)

                ax.set_title("Detected Hits (Time Domain)")
                ax.set_axis_off()
                st.pyplot(fig)

            # --- PER-HIT RESULTS ---
            st.subheader("📊 Per-Hit Predictions")

            good_hits = np.sum(preds == 0)
            bad_hits = np.sum(preds == 1)

            for i, p in enumerate(preds):
                label = "BAD" if p == 1 else "GOOD"
                st.write(f"Hit {i+1}: {label}")

            # --- DISTRIBUTION ---
            st.subheader("📈 Hit Distribution")

            st.write(f"GOOD hits: {good_hits}")
            st.write(f"BAD hits: {bad_hits}")

            st.bar_chart({
                "GOOD": good_hits,
                "BAD": bad_hits
            })

            # --- DETAILED GRAPHS ---
            c1, c2 = st.columns(2)

            with c1:
                st.caption("MFCC (Acoustic Fingerprint)")
                mfcc_vis = librosa.feature.mfcc(y=hits[0], sr=sr, n_mfcc=13)
                fig_m, ax_m = plt.subplots(figsize=(5, 2))
                librosa.display.specshow(mfcc_vis, ax=ax_m)
                st.pyplot(fig_m)

            with c2:
                st.caption("Frequency Energy (PSD)")
                fft = np.abs(np.fft.fft(hits[0]))**2
                freqs = np.fft.fftfreq(len(hits[0]), 1/sr)

                fig_f, ax_f = plt.subplots(figsize=(5, 2))
                ax_f.plot(freqs[:len(freqs)//2],
                          fft[:len(fft)//2],
                          color='purple')
                ax_f.set_xlim(0, 4000)
                st.pyplot(fig_f)
