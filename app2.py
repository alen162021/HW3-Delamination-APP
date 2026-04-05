import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

# --- SETTINGS & CONSTANTS ---
MODELS = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True, kernel='rbf')
}

# --- FEATURE EXTRACTION ---
def extract_features(signal, sr):
    # MFCCs (13 coefficients)
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13), axis=1)
    # PSD Mean
    psd = np.abs(np.fft.fft(signal))**2
    psd_mean = np.mean(psd)
    return np.hstack([mfcc, psd_mean])

# --- TRAINING PIPELINE (Part 2) ---
def train_and_evaluate(X, y, X_test_hw2=None, y_test_hw2=None):
    results = {}
    
    # Split 7:3 (Part 2, Task 3)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42
    )

    for name, model in MODELS.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        
        # Accuracy
        acc_train = accuracy_score(y_train, train_preds)
        acc_val = accuracy_score(y_val, val_preds)
        
        # Part 3: Robustness on HW2 Data
        acc_test = None
        cm_test = None
        if X_test_hw2 is not None:
            test_preds = model.predict(X_test_hw2)
            acc_test = accuracy_score(y_test_hw2, test_preds)
            cm_test = confusion_matrix(y_test_hw2, test_preds)

        results[name] = {
            "model": model,
            "acc_train": acc_train,
            "acc_val": acc_val,
            "acc_test": acc_test,
            "cm_train": confusion_matrix(y_train, train_preds),
            "cm_val": confusion_matrix(y_val, val_preds),
            "cm_test": cm_test
        }
    return results

# --- UI LOGIC ---
st.title("🔊 Composite Structural Health Monitor: HW3 Edition")
st.write("University of Houston | Smart Materials and Structures Lab")

# 1. DATA LOADING SECTION
st.header("Step 1: Load Datasets")
col1, col2 = st.columns(2)
with col1:
    hw3_files = st.file_uploader("Upload HW3 Data (Train/Val)", accept_multiple_files=True)
with col2:
    hw2_files = st.file_uploader("Upload HW2 Data (Independent Test)", accept_multiple_files=True)

if hw3_files:
    # This block assumes you have logic to extract features from all uploaded files
    # For brevity, let's assume 'X_hw3', 'y_hw3' are generated from your 'extract_features' function
    
    # [Placeholder for your file processing loop to build X and y matrices]
    # X_hw3 = np.array(all_features_hw3)
    # y_hw3 = np.array(all_labels_hw3)
    
    if st.button("🚀 Run Multi-Model Analysis"):
        # Run training
        # results = train_and_evaluate(X_hw3, y_hw3, X_hw2, y_hw2)
        
        st.divider()
        st.header("📊 Results Comparison")
        
        # --- ACCURACY TABLE (Part 3, Task 3) ---
        data = []
        for name, r in results.items():
            drop = r['acc_val'] - r['acc_test'] if r['acc_test'] else 0
            data.append({
                "Model": name,
                "Train Acc": f"{r['acc_train']:.2%}",
                "Val Acc": f"{r['acc_val']:.2%}",
                "Test (HW2) Acc": f"{r['acc_test']:.2%}" if r['acc_test'] else "N/A",
                "Accuracy Drop": f"{drop:.2%}"
            })
        st.table(pd.DataFrame(data))

        # --- CONFUSION MATRICES ---
        st.header("📈 Confusion Matrices")
        tabs = st.tabs(list(MODELS.keys()))
        
        for i, (name, r) in enumerate(results.items()):
            with tabs[i]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write("Training Set")
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay(r['cm_train']).plot(ax=ax, cmap='Blues')
                    st.pyplot(fig)
                with c2:
                    st.write("Validation Set")
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay(r['cm_val']).plot(ax=ax, cmap='Greens')
                    st.pyplot(fig)
                with c3:
                    if r['cm_test'] is not None:
                        st.write("HW2 (Unseen) Test")
                        fig, ax = plt.subplots()
                        ConfusionMatrixDisplay(r['cm_test']).plot(ax=ax, cmap='Oranges')
                        st.pyplot(fig)
