import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Delamination ML Lab", layout="wide", page_icon="🔊")

st.title("🔊 Delamination Detection & Machine Learning Lab")

# =========================
# LOAD AUDIO (FULL SAFE VERSION)
# =========================
def load_audio(file):
    try:
        # --- Try direct librosa (best case) ---
        signal, sr = librosa.load(file, sr=22050, backend="audioread")
        return signal, sr

    except Exception:
        try:
            # --- Save to temp file safely ---
            file.seek(0)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            # --- Try again with path ---
            signal, sr = librosa.load(tmp_path, sr=22050)
            return signal, sr

        except Exception as e:
            raise ValueError(f"Audio load failed for {file.name}: {str(e)}")


# =========================
# SPLIT HITS
# =========================
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
        start, end = seg[0] * hop_length, seg[-1] * hop_length
        hit = signal[start:end]

        if len(hit) > 200:
            hits.append(hit)
            boundaries.append((start, end))

    return hits, boundaries


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(signal, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13), axis=1)
    psd = np.mean(np.abs(np.fft.fft(signal))**2)
    return np.hstack([mfcc, psd])


# =========================
# BUILD DATASET
# =========================
def build_dataset(files):
    X, y = [], []

    for file in files:
        try:
            signal, sr = load_audio(file)
            hits, _ = split_hits(signal, sr)

            label = 1 if "_b" in file.name.lower() else 0

            for h in hits:
                X.append(extract_features(h, sr))
                y.append(label)

        except Exception as e:
            st.warning(f"Skipping {file.name}: {e}")

    if len(X) == 0:
        return np.array([]), np.array([])

    return np.array(X), np.array(y)


# =========================
# ANALYZE FILE
# =========================
def analyze_file(file, model):
    try:
        signal, sr = load_audio(file)
    except Exception as e:
        st.error(str(e))
        return None

    hits, boundaries = split_hits(signal, sr)

    if not hits:
        return None

    features = [extract_features(h, sr) for h in hits]
    preds = model.predict(features)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[:, 1]
        confidence = np.mean(probs)
    else:
        confidence = np.mean(preds)

    return signal, sr, hits, boundaries, preds, confidence


# =========================
# SIDEBAR (YOUR TEXT)
# =========================
with st.sidebar:
    st.header("🔍 Science Behind the Sound")

    st.subheader("Signal Processing")

    with st.expander("What is the Time Domain?"):
        st.write("It shows the sound's 'Heartbeat.' Healthy blocks ring longer; damaged blocks fade out fast due to internal friction.")

    with st.expander("What is the Frequency Graph?"):
        st.write("It shows the 'Pitch.' Delamination makes the block less stiff, shifting pitch lower.")

    with st.expander("What are MFCCs?"):
        st.write("The 'Acoustic Fingerprint.' Used by AI to recognize defects.")

    st.subheader("The 'Brain' (AI Algorithms)")

    with st.expander("KNN"):
        st.write("Majority vote from nearest neighbors.")

    with st.expander("Logistic Regression"):
        st.write("Probability-based classification.")

    with st.expander("Decision Tree"):
        st.write("Flowchart-style decision making.")

    with st.expander("SVM"):
        st.write("Draws a boundary between classes.")


# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📂 Data", "🤖 Training", "🧪 Testing"])


# =========================
# TAB 1: DATA
# =========================
with tab1:
    files = st.file_uploader("Upload Dataset (_g/_b)", accept_multiple_files=True)

    if files:
        X, y = build_dataset(files)

        if len(X) == 0:
            st.error("No valid data.")
        else:
            st.success(f"Samples: {len(X)}")
            st.session_state["X"] = X
            st.session_state["y"] = y


# =========================
# TAB 2: TRAINING
# =========================
with tab2:
    if "X" not in st.session_state:
        st.warning("Upload data first.")
    else:
        X = st.session_state["X"]
        y = st.session_state["y"]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(probability=True)
        }

        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, model.predict(X_train))
            val_acc = accuracy_score(y_val, model.predict(X_val))

            results.append((name, train_acc, val_acc))

        best_index = np.argmax([r[2] for r in results])

        st.session_state["models"] = models
        st.session_state["best_index"] = best_index


# =========================
# TAB 3: TESTING
# =========================
with tab3:
    if "models" not in st.session_state:
        st.warning("Train models first.")
    else:
        test_files = st.file_uploader("Upload Test Files", accept_multiple_files=True)

        if test_files:
            best_model = list(st.session_state["models"].values())[st.session_state["best_index"]]

            for file in test_files:
                with st.expander(file.name, expanded=True):

                    result = analyze_file(file, best_model)

                    if result is None:
                        st.warning("No hits detected.")
                        continue

                    signal, sr, hits, boundaries, preds, confidence = result

                    st.info(f"Hits detected: {len(hits)}")

                    # =========================
                    # TIME DOMAIN
                    # =========================
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.plot(signal, alpha=0.5)

                    for s, e in boundaries:
                        ax.axvspan(s, e, alpha=0.3)

                    ax.set_title("Time Domain (Hits)")
                    ax.set_axis_off()
                    st.pyplot(fig)

                    # =========================
                    # FREQUENCY DOMAIN
                    # =========================
                    fft = np.abs(np.fft.fft(signal))**2
                    freqs = np.fft.fftfreq(len(signal), 1/sr)

                    fig2, ax2 = plt.subplots(figsize=(6, 2))
                    ax2.plot(freqs[:len(freqs)//2], fft[:len(fft)//2])
                    ax2.set_xlim(0, 4000)
                    ax2.set_title("Frequency Domain")
                    st.pyplot(fig2)

                    # =========================
                    # RESULT
                    # =========================
                    if confidence > 0.5:
                        st.error(f"DEFECT ({confidence*100:.1f}%)")
                    else:
                        st.success(f"HEALTHY ({(1-confidence)*100:.1f}%)")
