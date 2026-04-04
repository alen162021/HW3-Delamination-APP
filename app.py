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

# --- PAGE CONFIG ---
st.set_page_config(page_title="HW3 Delamination ML System", layout="wide")

st.title("🔊 HW3: Automated Delamination ML Pipeline")

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

    energy = librosa.feature.rms(y=signal,
                                 frame_length=frame_length,
                                 hop_length=hop_length)[0]

    indices = np.where(energy > np.mean(energy) * 1.5)[0]

    if len(indices) == 0:
        return []

    segments = np.split(indices, np.where(np.diff(indices) > 2)[0] + 1)

    hits = []
    for seg in segments:
        start = seg[0] * hop_length
        end = seg[-1] * hop_length
        hit = signal[start:end]

        if len(hit) > 200:
            hits.append(hit)

    return hits

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
        hits = split_hits(signal, sr)

        label = 1 if "_b" in file.name.lower() else 0

        for h in hits:
            features = extract_features(h, sr)
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

# --- MODEL TRAINING ---
def train_models(X_train, y_train):
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models

# --- EVALUATION ---
def evaluate_model(model, X, y):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)
    return acc, cm

# --- UI ---
st.header("📂 Upload HW3 Dataset (Training + Validation)")
train_files = st.file_uploader(
    "Upload HW3 audio files (multiple allowed)",
    type=["wav", "m4a"],
    accept_multiple_files=True
)

st.header("📂 Upload HW2 Dataset (Unseen Test Data)")
test_files = st.file_uploader(
    "Upload HW2 audio files",
    type=["wav", "m4a"],
    accept_multiple_files=True
)

if train_files:

    st.subheader("🔄 Building Dataset...")
    X, y = build_dataset(train_files)

    st.write(f"Total samples: {len(X)}")

    # --- SPLIT ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42
    )

    st.subheader("🤖 Training Models...")
    models = train_models(X_train, y_train)

    results = []

    st.subheader("📊 Results (Training & Validation)")

    for name, model in models.items():

        train_acc, train_cm = evaluate_model(model, X_train, y_train)
        val_acc, val_cm = evaluate_model(model, X_val, y_val)

        st.markdown(f"### {name}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Training Accuracy:", round(train_acc, 4))
            st.write("Confusion Matrix:")
            st.write(train_cm)

        with col2:
            st.write("Validation Accuracy:", round(val_acc, 4))
            st.write("Confusion Matrix:")
            st.write(val_cm)

        results.append({
            "Model": name,
            "Train Acc": train_acc,
            "Val Acc": val_acc
        })

    # --- PART 3: TEST DATA ---
    if test_files:

        st.subheader("🧪 Testing on HW2 (Unseen Data)")

        X_test, y_test = build_dataset(test_files)

        for i, (name, model) in enumerate(models.items()):

            test_acc, test_cm = evaluate_model(model, X_test, y_test)

            st.markdown(f"### {name} (Test Results)")

            st.write("Test Accuracy:", round(test_acc, 4))
            st.write("Confusion Matrix:")
            st.write(test_cm)

            # --- ROBUSTNESS ---
            val_acc = results[i]["Val Acc"]
            drop = val_acc - test_acc

            st.write(f"Accuracy Drop (Val → Test): {round(drop,4)}")

            results[i]["Test Acc"] = test_acc
            results[i]["Drop"] = drop

    # --- FINAL TABLE ---
    st.subheader("📋 Model Comparison Table")

    st.dataframe(results)
