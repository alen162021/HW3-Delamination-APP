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
st.write("This app performs delamination analysis: signal processing, feature extraction, model training, and robustness testing.")

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

# --- SIDEBAR (EDUCATIONAL) ---
with st.sidebar:
    st.header("🔍 Learn What’s Happening")

    with st.expander("Time Domain (Waveform)"):
        st.write("Shows how sound amplitude changes over time. Damaged structures lose energy faster.")

    with st.expander("Frequency Domain"):
        st.write("Shows energy distribution. Damage shifts energy to lower frequencies.")

    with st.expander("MFCC Features"):
        st.write("Compact representation of sound used by ML models.")

    with st.expander("Machine Learning Models"):
        st.write("""
        - KNN: Uses nearest neighbors
        - Decision Tree: Rule-based splits
        - Logistic Regression: Linear classifier
        - SVM: Finds optimal boundary
        """)

# --- FILE UPLOAD ---
st.header("📂 Upload Dataset (Training + Validation)")
train_files = st.file_uploader("Upload labeled files (_g / _b)", accept_multiple_files=True)

st.header("📂 Upload Dataset (Unseen Test)")
test_files = st.file_uploader("Upload test dataset", accept_multiple_files=True)

# --- PROCESS ---
if train_files:

    st.subheader("🔄 Step 1: Building Dataset")
    X, y = build_dataset(train_files)

    st.write(f"Total samples extracted: {len(X)}")

    st.info("Each impact is converted into features (MFCC + PSD) and labeled as GOOD (0) or BAD (1).")

    # --- VISUALIZE DATASET ---
    st.subheader("📊 Dataset Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Label Distribution")
        unique, counts = np.unique(y, return_counts=True)
        st.bar_chart(dict(zip(unique, counts)))

    with col2:
        st.write("Feature Spread (First 2 MFCCs)")
        if X.shape[1] >= 2:
            fig, ax = plt.subplots()
            ax.scatter(X[:, 0], X[:, 1], c=y)
            ax.set_xlabel("MFCC 1")
            ax.set_ylabel("MFCC 2")
            st.pyplot(fig)

    # --- SPLIT ---
    st.subheader("✂️ Step 2: Train/Validation Split")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    st.info("Dataset is split into 70% training and 30% validation.")

    # --- TRAIN MODELS ---
    st.subheader("🤖 Step 3: Training Models")

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        st.markdown(f"### {name}")

        colA, colB = st.columns(2)

        with colA:
            st.write("Training Accuracy:", round(train_acc, 4))
            st.write(confusion_matrix(y_train, train_pred))

        with colB:
            st.write("Validation Accuracy:", round(val_acc, 4))
            st.write(confusion_matrix(y_val, val_pred))

        results.append({
            "Model": name,
            "Train": train_acc,
            "Validation": val_acc
        })

    # --- TEST (HW2) ---
    if test_files:

        st.subheader("🧪 Step 4: Testing on Unseen Data (HW2)")

        X_test, y_test = build_dataset(test_files)

        for i, (name, model) in enumerate(models.items()):

            test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)

            st.markdown(f"### {name}")

            st.write("Test Accuracy:", round(test_acc, 4))
            st.write(confusion_matrix(y_test, test_pred))

            drop = results[i]["Validation"] - test_acc
            st.write(f"Accuracy Drop (Robustness): {round(drop,4)}")

            results[i]["Test"] = test_acc
            results[i]["Drop"] = drop

    # --- FINAL TABLE ---
    st.subheader("📋 Final Comparison")

    st.dataframe(results)
