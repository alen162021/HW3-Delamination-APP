import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# --- PAGE CONFIG ---
st.set_page_config(page_title="HW3: Delamination ML Lab", layout="wide")
st.title("🔊 HW3: Composite Delamination Detection")
st.markdown("### KNN, DT, LR, and SVM Model Analysis")

# --- UTILITY FUNCTIONS ---
def extract_features(signal, sr):
    # PSD Calculation
    psd = np.mean(np.abs(np.fft.fft(signal))**2)
    # MFCC Calculation (13 coefficients)
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13), axis=1)
    return np.hstack([mfcc, psd])

def process_files(files):
    X, y = [], []
    for f in files:
        # Load audio
        sig, sr = librosa.load(f, sr=22050)
        # Simple energy-based hit detection
        # Note: In a real lab, you'd use your split_hits logic here
        # For brevity, we treat the file as the sample or extract the main hit
        feat = extract_features(sig, sr)
        X.append(feat)
        # Labeling based on filename (Requirement: _g = 0, _b = 1)
        y.append(1 if "_b" in f.name.lower() else 0)
    return np.array(X), np.array(y)

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Healthy', 'Damage'], yticklabels=['Healthy', 'Damage'])
    ax.set_title(title)
    return fig

# --- TABS FOR HOMEWORK PARTS ---
tab1, tab2, tab3 = st.tabs(["📁 Part 1 & 2: Training", "🧪 Part 3: Robustness (Unseen)", "📊 Summary Report"])

with tab1:
    st.header("Model Training & Validation")
    train_files = st.file_uploader("Upload HW3 Dataset (_g and _b files)", accept_multiple_files=True, key="train")
    
    if train_files:
        X, y = process_files(train_files)
        # Part 2.3: 7:3 Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
        
        models = {
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "Decision Tree": DecisionTreeClassifier(),
            "Logistic Regression": LogisticRegression(),
            "SVM": SVC(kernel='linear', probability=True)
        }
        
        col1, col2 = st.columns(2)
        metrics = []

        for name, clf in models.items():
            clf.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            val_acc = accuracy_score(y_val, clf.predict(X_val))
            metrics.append({"Model": name, "Train Acc": train_acc, "Val Acc": val_acc})
            
            with st.expander(f"Confusion Matrices: {name}"):
                c1, c2 = st.columns(2)
                c1.pyplot(plot_confusion_matrix(y_train, clf.predict(X_train), "Training"))
                c2.pyplot(plot_confusion_matrix(y_val, clf.predict(X_val), "Validation"))
        
        st.session_state['trained_models'] = models
        st.table(pd.DataFrame(metrics))

with tab3:
    st.header("HW3 Performance Summary")
    if 'trained_models' in st.session_state:
        st.info("Use the data above to fill your Part 4 Report Table.")
        # Logic to compare accuracy drop would go here
