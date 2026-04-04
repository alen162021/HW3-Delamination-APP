import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tempfile
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# --- PAGE CONFIG ---
st.set_page_config(page_title="HDelamination ML Lab", layout="wide", page_icon="🔊")

# --- STYLE ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- CORE FUNCTIONS ---

def extract_features(signal, sr):
    """Part 2.4: Feature extraction (PSD and MFCC)"""
    # MFCCs: Acoustic Fingerprint
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13), axis=1)
    # PSD: Power Spectral Density (Energy)
    psd = np.mean(np.abs(np.fft.fft(signal))**2)
    return np.hstack([mfcc, psd])

def process_uploaded_files(uploaded_files):
    """Handles LibsndfileError by saving to temp disk first"""
    X, y = [], []
    for f in uploaded_files:
        try:
            # Create a real file path for librosa to read
            suffix = os.path.splitext(f.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.getbuffer())
                tmp_path = tmp.name

            # Load audio (22050 Hz is standard for percussion)
            sig, sr = librosa.load(tmp_path, sr=22050)
            os.remove(tmp_path) # Clean up disk

            # Labeling based on assignment naming convention (_g vs _b)
            label = 1 if "_b" in f.name.lower() else 0
            
            # Feature extraction
            feat = extract_features(sig, sr)
            X.append(feat)
            y.append(label)
        except Exception as e:
            st.error(f"Could not process {f.name}: {e}")
    return np.array(X), np.array(y)

def plot_cm(y_true, y_pred, title):
    """Generates the Confusion Matrix required for Part 4"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', ax=ax, 
                xticklabels=['Healthy', 'Damage'], yticklabels=['Healthy', 'Damage'])
    ax.set_title(title)
    plt.tight_layout()
    return fig

# --- UI LAYOUT ---
st.title("🔊 Composite Structural Health Monitoring")
st.caption("Smart Materials and Structures Lab | University of Houston")

tab1, tab2, tab3 = st.tabs(["📂 Data Collection", "🤖 Model Training", "🧪 Robustness Test"])

# --- TAB 1: DATA LOADING ---
with tab1:
    st.header("Part 1: Upload HW3 Dataset")
    st.info("Ensure files are named like 's_1_g.wav' (Good) or 's_2_b.wav' (Bad).")
    files = st.file_uploader("Upload Audio Files", accept_multiple_files=True, type=['wav', 'm4a'])
    
    if files:
        with st.spinner("Processing signals..."):
            X, y = process_uploaded_files(files)
            if len(X) > 0:
                st.success(f"Successfully processed {len(X)} samples.")
                st.session_state['X'] = X
                st.session_state['y'] = y
            else:
                st.error("No valid audio features extracted.")

# --- TAB 2: TRAINING ---
with tab2:
    if 'X' not in st.session_state:
        st.warning("Please upload data in Tab 1 first.")
    else:
        st.header("Part 2: Algorithm Performance")
        
        # Part 2.3: 70/30 Split
        X_train, X_val, y_train, y_val = train_test_split(
            st.session_state['X'], st.session_state['y'], test_size=0.3, random_state=42, shuffle=True
        )

        models = {
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(kernel='linear', probability=True)
        }

        results_list = []
        trained_clfs = {}

        for name, clf in models.items():
            clf.fit(X_train, y_train)
            trained_clfs[name] = clf
            
            y_train_pred = clf.predict(X_train)
            y_val_pred = clf.predict(X_val)
            
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            
            results_list.append({"Model": name, "Train Acc": train_acc, "Val Acc": val_acc})
            
            with st.expander(f"Confusion Matrices: {name}"):
                c1, c2 = st.columns(2)
                c1.pyplot(plot_cm(y_train, y_train_pred, f"{name} - Training"))
                c2.pyplot(plot_cm(y_val, y_val_pred, f"{name} - Validation"))

        st.session_state['trained_clfs'] = trained_clfs
        st.subheader("Comparison Table")
        st.table(pd.DataFrame(results_list))

# --- TAB 3: ROBUSTNESS ---
with tab3:
    st.header("Part 3: Unseen Data (Dataset)")
    st.write("Upload your data here to test model robustness.")
    hw2_files = st.file_uploader("Upload Files", accept_multiple_files=True, type=['wav', 'm4a'])
    
    if hw2_files and 'trained_clfs' in st.session_state:
        X_test, y_test = process_uploaded_files(hw2_files)
        
        if len(X_test) > 0:
            robustness_results = []
            for name, clf in st.session_state['trained_clfs'].items():
                y_test_pred = clf.predict(X_test)
                test_acc = accuracy_score(y_test, y_test_pred)
                robustness_results.append({"Model": name, "HW2 Test Acc": test_acc})
                
                with st.expander(f"Robustness Matrix: {name}"):
                    st.pyplot(plot_cm(y_test, y_test_pred, f"{name} - Unseen HW2 Data"))
            
            st.subheader("Robustness Comparison")
            st.table(pd.DataFrame(robustness_results))
        else:
            st.error("No valid features in HW2 data.")
    elif not hw2_files:
        st.info("Waiting for HW2 data upload...")
