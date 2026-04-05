import streamlit as st
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. FEATURE EXTRACTION ENGINE ---
def extract_features_from_hits(hits, sr):
    feature_list = []
    for h in hits:
        # MFCCs (13 features)
        mfccs = np.mean(librosa.feature.mfcc(y=h, sr=sr, n_mfcc=13), axis=1)
        # PSD (1 feature - Mean Power)
        psd = np.mean(np.abs(np.fft.fft(h))**2)
        # Combine into a 14-element vector
        feature_list.append(np.hstack([mfccs, psd]))
    return np.array(feature_list)

# --- 2. THE TRAINING & EVALUATION INTERFACE ---
st.title("🚀 Model Training & Robustness Evaluator")

tab1, tab2 = st.tabs(["Live Detector", "Train & Compare (HW3 vs HW2)"])

with tab2:
    st.header("Part 2 & 3: Multi-Model Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        hw3_files = st.file_uploader("Upload HW3 Data (Training/Val)", accept_multiple_files=True, key="hw3")
    with col2:
        hw2_files = st.file_uploader("Upload HW2 Data (Independent Test)", accept_multiple_files=True, key="hw2")

    if st.button("Run Training and Robustness Test"):
        # Placeholder for data processing
        # In a real scenario, you'd loop through files, extract hits, and label them
        # X_hw3, y_hw3 = process_files(hw3_files)
        # X_hw2, y_hw2 = process_files(hw2_files)
        
        # --- Task 2.5: Define Models ---
        models = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(kernel='rbf', probability=True)
        }

        # Task 2.3: 70/30 Split
        # X_train, X_val, y_train, y_val = train_test_split(X_hw3, y_hw3, test_size=0.3)

        results = []
        
        for name, clf in models.items():
            # clf.fit(X_train, y_train)
            
            # --- CALCULATE ACCURACIES ---
            # acc_train = accuracy_score(y_train, clf.predict(X_train))
            # acc_val = accuracy_score(y_val, clf.predict(X_val))
            # acc_test = accuracy_score(y_hw2, clf.predict(X_hw2)) # Part 3
            
            results.append({
                "Model": name,
                "Train Acc": "0.98", # Example placeholder
                "Val Acc": "0.92",
                "HW2 Test Acc": "0.85",
                "Drop": "0.07"
            })
            
            # --- PLOT CONFUSION MATRICES ---
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            # sns.heatmap(confusion_matrix(y_val, p_val), annot=True, ax=ax[0])
            # sns.heatmap(confusion_matrix(y_hw2, p_test), annot=True, ax=ax[1])
            st.write(f"### {name} Confusion Matrices")
            st.pyplot(fig)

        # --- Task 3.3: Comparison Table ---
        st.subheader("Final Performance Comparison")
        st.table(pd.DataFrame(results))
