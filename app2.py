import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Acoustic Structural Health Monitor", layout="wide", page_icon="🛡️")

# --- CORE UTILITIES ---

def extract_features(signal, sr):
    """
    Extracts 13 MFCCs and the Mean Power Spectral Density (PSD).
    Returns a combined feature vector of length 14.
    """
    # MFCC extraction
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13), axis=1)
    
    # PSD calculation
    psd = np.abs(np.fft.fft(signal))**2
    psd_mean = np.mean(psd)
    
    return np.hstack([mfcc, psd_mean])

def process_audio_files(uploaded_files, label):
    """
    Saves uploaded bytes to a temp file, loads via librosa, 
    detects individual hits, and extracts features.
    """
    features_list = []
    
    for file in uploaded_files:
        # Create a temporary file to handle .m4a/.wav compatibility
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        try:
            # Load audio from the path (required for stable decoding of compressed formats)
            signal, sr = librosa.load(tmp_path, sr=None)
            
            # 1. Noise Removal: Trim silence/low-energy parts
            yt, _ = librosa.effects.trim(signal, top_db=20)
            
            # 2. Multi-Hit Splitting: Detect onsets of percussion hits
            onsets = librosa.onset.onset_detect(y=yt, sr=sr, units='samples', backtrack=True)
            
            for i in range(len(onsets)):
                start = onsets[i]
                end = onsets[i+1] if i+1 < len(onsets) else len(yt)
                hit = yt[start:end]
                
                # Feature extraction (ensure hit is long enough for FFT)
                if len(hit) > 2048:
                    features_list.append(extract_features(hit, sr))
        
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
        
        finally:
            # Clean up the temporary file from the server
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    if not features_list:
        return np.empty((0, 14)), np.empty((0,))
        
    return np.array(features_list), np.full(len(features_list), label)

# --- UI LAYOUT ---

st.title("🛡️ Composite Structural Health Monitor")
st.markdown("""
Evaluate the structural integrity of composite materials using acoustic percussion analysis. 
This system compares multiple machine learning architectures to identify the most robust detection model.
""")

# Setup Tabs for Data Input
tab_train, tab_test = st.tabs(["📁 Training & Validation Data", "📁 Independent Robustness Test"])

with tab_train:
    st.subheader("Reference Data")
    col1, col2 = st.columns(2)
    with col1:
        h_files = st.file_uploader("Upload Healthy Samples (A)", accept_multiple_files=
