import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# --- PAGE CONFIG ---
st.set_page_config(page_title="Delamination ML Lab", layout="wide", page_icon="🔊")

st.title("🔊 Delamination Detection & Machine Learning Lab")

st.caption("""
This project was partially sponsored by a Teaching Innovation Program (TIP) grant, 
Smart Materials and Structures Laboratory (SMSL), and Artificial Intelligent Laboratory 
for Monitoring and Inspection, University of Houston.
""")

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
    frame_length, hop_length = int(0.02 * sr), int(0.01 * sr)

    energy = librosa.feature.rms(y=signal,
                                 frame_length=frame_length,
                                 hop_length=hop_length)[0]

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

# --- FEATURE EXTRACTION ---
def extract_features(signal, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13), axis=1)
    psd = np.mean(np.abs(np.fft.fft(signal))**2)
    return np.hstack([mfcc, psd])

# --- DATASET BUILDER ---
def build_dataset(files):
    X, y = [], []

    for file in files:
        signal, sr = load_audio(file)
        hits, _ = split_hits(signal, sr)

        label = 1 if "_b" in file.name.lower() else 0

        for h in hits:
            X.append(extract_features(h, sr))
            y.append(label)

    return np.array(X), np.array(y)

# --- HEATMAP FUNCTION ---
def plot_conf_matrix(cm, title):
    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Learn the Concepts")

    st.markdown("""
    **Time Domain:** Shows how sound decays over time  
    **Frequency Domain:** Shows energy distribution  
    **MFCC:** Acoustic fingerprint used for ML  
    """)

    st.markdown("""
    **Models Used:**
    - KNN
    - Decision Tree
    - Logistic Regression
    - SVM
    """)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📂 Data", "🤖 Training", "🧪 Testing"])

# =======================
# TAB 1: DATA
# =======================
with tab1:
    st.header("Upload Dataset")

    train_files = st.file_uploader(
        "Upload HW3 Dataset (_g / _b)",
        accept_multiple_files=True
    )

    if train_files:
        X, y = build_dataset(train_files)

        st.success(f"Extracted {len(X)} samples")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Label Distribution")
            unique, counts = np.unique(y, return_counts=True)
            st.bar_chart(dict(zip(unique, counts)))

        with col2:
            st.subheader("Feature Scatter (MFCC1 vs MFCC2)")
            if X.shape[1] >= 2:
                fig, ax = plt.subplots()
                ax.scatter(X[:, 0], X[:, 1], c=y)
                st.pyplot(fig)

        st.session_state["X"] = X
        st.session_state["y"] = y

# =======================
# TAB 2: TRAINING
# =======================
with tab2:
    if "X" not in st.session_state:
        st.warning("Upload dataset first.")
    else:
        X = st.session_state["X"]
        y = st.session_state["y"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC()
        }

        results = []

        for name, model in models.items():
            st.subheader(name)

            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)

            col1, col2 = st.columns(2)

            with col1:
                st.write("Training Accuracy:", round(train_acc, 4))
                plot_conf_matrix(confusion_matrix(y_train, train_pred), "Train CM")

            with col2:
                st.write("Validation Accuracy:", round(val_acc, 4))
                plot_conf_matrix(confusion_matrix(y_val, val_pred), "Validation CM")

            results.append({
                "Model": name,
                "Validation": val_acc
            })

        st.session_state["models"] = models
        st.session_state["results"] = results

# =======================
# TAB 3: TESTING
# =======================
with tab3:
    if "models" not in st.session_state:
        st.warning("Train models first.")
    else:
        test_files = st.file_uploader(
            "Upload Dataset",
            accept_multiple_files=True
        )

        if test_files:
            X_test, y_test = build_dataset(test_files)

            st.subheader("Test Data Visualization")

            fig, ax = plt.subplots()
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
            st.pyplot(fig)

            for i, (name, model) in enumerate(st.session_state["models"].items()):
                st.subheader(name)

                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)

                st.write("Test Accuracy:", round(acc, 4))
                plot_conf_matrix(confusion_matrix(y_test, pred), "Test CM")

                drop = st.session_state["results"][i]["Validation"] - acc
                st.write("Accuracy Drop:", round(drop, 4))
