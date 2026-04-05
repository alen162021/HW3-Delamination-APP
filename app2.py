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
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Advanced Composite Monitor", layout="wide", page_icon="🛡️")

# --- CORE FUNCTIONS ---

def extract_features(signal, sr):
    """Extracts MFCCs and PSD Mean as a combined feature vector."""
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13), axis=1)
    psd = np.abs(np.fft.fft(signal))**2
    psd_mean = np.mean(psd)
    return np.hstack([mfcc, psd_mean])

def process_audio_files(uploaded_files, label):
    """Processes multiple files into a feature matrix X and label vector y."""
    features_list = []
    for file in uploaded_files:
        # Load audio (handles .m4a and .wav)
        signal, sr = librosa.load(io.BytesIO(file.read()), sr=None)
        
        # Simple Noise Removal / Silence Trimming
        yt, _ = librosa.effects.trim(signal, top_db=20)
        
        # Split hits (Basic onset detection)
        onsets = librosa.onset.onset_detect(y=yt, sr=sr, units='samples')
        for i in range(len(onsets)):
            start = onsets[i]
            end = onsets[i+1] if i+1 < len(onsets) else len(yt)
            hit = yt[start:end]
            
            # Ensure hit is long enough for feature extraction
            if len(hit) > 2048:
                features_list.append(extract_features(hit, sr))
                
    X = np.array(features_list)
    y = np.full(len(features_list), label)
    return X, y

# --- SIDEBAR & MODELS ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    st.info("This system evaluates four classification architectures to determine the most robust detector.")
    
    st.divider()
    st.caption("Smart Materials and Structures Lab")
    st.caption("University of Houston")

# --- MAIN UI ---
st.title("🛡️ Composite Structural Health Monitor")
st.write("Upload acoustic percussion data to train and evaluate delamination detection models.")

# --- DATA UPLOAD TABS ---
tab1, tab2 = st.tabs(["📁 Training Data (Current)", "📁 Robustness Test (Independent)"])

with tab1:
    col_h, col_d = st.columns(2)
    with col_h:
        healthy_files = st.file_uploader("Upload Healthy Samples", accept_multiple_files=True, key="h1")
    with col_d:
        damaged_files = st.file_uploader("Upload Damaged Samples", accept_multiple_files=True, key="d1")

with tab2:
    st.write("Upload a separate, unseen dataset to test model generalization.")
    col_h2, col_d2 = st.columns(2)
    with col_h2:
        test_healthy = st.file_uploader("Upload Unseen Healthy", accept_multiple_files=True, key="h2")
    with col_d2:
        test_damaged = st.file_uploader("Upload Unseen Damaged", accept_multiple_files=True, key="d2")

# --- EXECUTION ---
if st.button("🚀 Run Multi-Model Analysis", type="primary"):
    if not (healthy_files and damaged_files):
        st.warning("Please upload both Healthy and Damaged samples for training.")
    else:
        with st.spinner("Processing signals and training models..."):
            # 1. Prepare Training/Validation Data
            X_h, y_h = process_audio_files(healthy_files, 0)
            X_d, y_d = process_audio_files(damaged_files, 1)
            X_main = np.vstack([X_h, X_d])
            y_main = np.hstack([y_h, y_d])
            
            # 2. Prepare Independent Test Data (if provided)
            X_test_unseen, y_test_unseen = None, None
            if test_healthy and test_damaged:
                X_th, y_th = process_audio_files(test_healthy, 0)
                X_td, y_td = process_audio_files(test_damaged, 1)
                X_test_unseen = np.vstack([X_th, X_td])
                y_test_unseen = np.hstack([y_th, y_td])

            # 3. Train/Val Split (70/30)
            X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=0.3, shuffle=True, random_state=42)

            # 4. Model Loop
            models = {
                "KNN": KNeighborsClassifier(n_neighbors=3),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "SVM": SVC(probability=True)
            }

            results_data = []
            
            st.divider()
            st.header("📊 Performance Metrics")
            
            # Define columns for Confusion Matrices
            cm_cols = st.columns(len(models))

            for idx, (name, clf) in enumerate(models.items()):
                # Fit
                clf.fit(X_train, y_train)
                
                # Accuracies
                acc_train = accuracy_score(y_train, clf.predict(X_train))
                acc_val = accuracy_score(y_val, clf.predict(X_val))
                
                acc_test = "N/A"
                drop = "N/A"
                
                if X_test_unseen is not None:
                    y_pred_unseen = clf.predict(X_test_unseen)
                    acc_test_val = accuracy_score(y_test_unseen, y_pred_unseen)
                    acc_test = f"{acc_test_val:.2%}"
                    drop = f"{(acc_val - acc_test_val):.2%}"
                
                results_data.append({
                    "Model": name,
                    "Training Acc": f"{acc_train:.2%}",
                    "Validation Acc": f"{acc_val:.2%}",
                    "Unseen Test Acc": acc_test,
                    "Robustness Drop": drop
                })

                # Plot Confusion Matrix in appropriate column
                with cm_cols[idx]:
                    st.subheader(name)
                    cm = confusion_matrix(y_val, clf.predict(X_val))
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False, cmap='Blues')
                    plt.title("Val Set CM")
                    st.pyplot(fig)

            # 5. Display Summary Table
            st.subheader("Model Accuracy Comparison")
            st.table(pd.DataFrame(results_data))
            
            st.success("Analysis Complete. Review the 'Robustness Drop' to find the best generalizing model.")
