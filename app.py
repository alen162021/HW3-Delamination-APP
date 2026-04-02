import streamlit as st
import zipfile, os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

st.title("Delamination Detection App")

uploaded_file = st.file_uploader("Upload Dataset ZIP", type="zip")

if uploaded_file:
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall("data")

    st.success("Data Loaded!")

    def get_label(filename):
        return 0 if "_g" in filename else 1

    def load_audio(path):
        return librosa.load(path, sr=22050)

    def extract_features(signal, sr):
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        fft = np.abs(np.fft.fft(signal))**2
        psd = np.mean(fft)

        return np.hstack([mfcc_mean, psd])

    def build_dataset(folder):
        X, y = [], []
        for file in os.listdir(folder):
            if file.endswith(".m4a") or file.endswith(".wav"):
                signal, sr = load_audio(os.path.join(folder, file))
                X.append(extract_features(signal, sr))
                y.append(get_label(file))
        return np.array(X), np.array(y)

    X, y = [], []
    for i in range(1,6):
        Xi, yi = build_dataset(f"data/Set {i}")
        X.extend(Xi)
        y.extend(yi)

    X = np.array(X)
    y = np.array(y)

    X_test, y_test = build_dataset("data/Main_Test_Set")

    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3)

    models = {
        "KNN": KNeighborsClassifier(),
        "DT": DecisionTreeClassifier(),
        "LR": LogisticRegression(max_iter=1000),
        "SVM": SVC()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        st.write(f"{name} Test Accuracy: {acc:.3f}")
