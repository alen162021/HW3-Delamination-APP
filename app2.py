import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import soundfile as sf

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

# =========================
# FUNCTIONS (FIXED AUDIO LOADING)
# =========================
def load_audio(file):
    suffix = "." + file.name.split(".")[-1]

    file.seek(0)  # 🔥 CRITICAL FIX

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp.flush()
        tmp_path = tmp.name

    try:
        signal, sr = librosa.load(tmp_path, sr=22050)
    except Exception:
        try:
            signal, sr = librosa.load(tmp_path, sr=22050, backend="audioread")
        except Exception:
            signal, sr = sf.read(tmp_path)
            if len(signal.shape) > 1:
                signal = np.mean(signal, axis=1)

    return signal, sr

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

def extract_features(signal, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13), axis=1)
    psd = np.mean(np.abs(np.fft.fft(signal))**2)
    return np.hstack([mfcc, psd])

def build_dataset(files):
    X, y = [], []

    for file in files:
        try:
            signal, sr = load_audio(file)
        except:
            st.warning(f"Skipping corrupted file: {file.name}")
            continue

        hits, _ = split_hits(signal, sr)
        label = 1 if "_b" in file.name.lower() else 0

        for h in hits:
            X.append(extract_features(h, sr))
            y.append(label)

    return np.array(X), np.array(y)

def plot_conf_matrix(cm, title):
    fig, ax = plt.subplots()
    ax.imshow(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

def analyze_file(file, model):
    try:
        signal, sr = load_audio(file)
    except:
        return None

    hits, boundaries = split_hits(signal, sr)

    if not hits:
        return None

    features = [extract_features(h, sr) for h in hits]
    preds = model.predict(features)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[:, 1]
        confidence = np.mean(probs)
    else:
        confidence = np.mean(preds)

    return signal, sr, hits, boundaries, preds, confidence

# =========================
# SIDEBAR (UNCHANGED)
# =========================
with st.sidebar:
    st.header("🔍 Science Behind the Sound")

    st.subheader("Signal Processing")
    with st.expander("What is the Time Domain?"):
        st.write("""
        **Observation:** Healthy blocks ring longer; damaged fade faster.
        
        **The Physics:** This is a plot of **Amplitude vs. Time**. In structural health monitoring, we look at the 'damping' ratio. A healthy composite plate is stiff and elastic, allowing energy to resonate. Delamination introduces internal friction and air gaps that absorb energy, causing the signal to decay (attenuate) much quicker.
        """)

    with st.expander("Frequency Graph (FFT)"):
        st.write("""
        **Observation:** Damage shifts resonance energy to lower frequencies.
        
        **The Physics:** Using the **Fast Fourier Transform (FFT)**, we move from time to the frequency spectrum. Because delamination reduces the effective stiffness ($k$) of the material, the natural resonance frequencies drop. A shift to the left in the primary peaks is a classic indicator of internal structural failure.
        """)
    
    with st.expander("Power Spectral Density (PSD)"):
        st.write("""
        **Observation:** Shows energy strength and distribution across frequencies.
        
        **The Physics:** While FFT shows individual peaks, **PSD** (calculated via Welch's method) provides a cleaner, averaged power estimate. It helps distinguish between true structural resonance and random background noise. Lower power in high-frequency bands often suggests the material can no longer support high-frequency vibrations due to damage.
        """)
    
    with st.expander("MFCC (Acoustic Fingerprint)"):
        st.write("""
        **Observation:** The 'Acoustic Fingerprint' used for Machine Learning.
        
        **The Physics:** **Mel-Frequency Cepstral Coefficients** compress the complex audio signal into 13-14 key features that mimic how human hearing perceives sound. By extracting these coefficients, we create a mathematical signature of the 'hit' that models like **SVM** or **KNN** can use to classify the health of the cell.
        """)
        
    with st.expander("1. K-Nearest Neighbors (KNN)"):
        st.write("""
        **How it works:** It looks at the '14-dimensional' fingerprint (MFCC + PSD) of a new hit and finds the 5 most similar sounds in the training data.
        **In this project:** If the closest 5 hits were mostly 'Damaged,' it classifies the new hit as Damaged. It is simple but sensitive to noisy recordings.
        """)
    
    with st.expander("2. Decision Tree (DT)"):
        st.write("""
        **How it works:** A flowchart of 'if-then' rules (e.g., *Is the PSD mean < X? Is the 1st MFCC > Y?*).
        **In this project:** It is very fast and easy to interpret, but prone to **overfitting**, meaning it might "memorize" the specific noise of your HW3 dataset rather than learning general rules.
        """)
    
    with st.expander("3. Logistic Regression (LR)"):
        st.write("""
        **How it works:** A statistical model that calculates the probability (0 to 1) of a sample being 'Healthy' or 'Damaged' based on a weighted sum of features.
        **In this project:** It provides a clear "decision boundary." It is often more robust than a Decision Tree because it doesn't over-react to single outlier hits.
        """)
    
    with st.expander("4. Support Vector Machine (SVM)"):
        st.write("""
        **How it works:** It finds the optimal "gap" or hyperplane that separates the Healthy data points from the Damaged ones in a high-dimensional space.
        **In this project:** Using a **Radial Basis Function (RBF)** kernel allows it to find complex patterns. It is typically the most robust model for unseen data (Part 3).
        """)
    

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📂 Data", "🤖 Training", "🧪 Testing"])

# =========================
# DATA TAB
# =========================
with tab1:
    train_files = st.file_uploader("Upload Training Dataset (_g/_b)", accept_multiple_files=True)

    if train_files:
        X, y = build_dataset(train_files)

        st.success(f"Samples: {len(X)}")

        col1, col2 = st.columns(2)

        with col1:
            unique, counts = np.unique(y, return_counts=True)
            st.bar_chart(dict(zip(unique, counts)))

        with col2:
            fig, ax = plt.subplots()
            ax.scatter(X[:, 0], X[:, 1], c=y)
            st.pyplot(fig)

        st.session_state["X"] = X
        st.session_state["y"] = y

# =========================
# TRAINING TAB
# =========================
with tab2:
    if "X" not in st.session_state:
        st.warning("Upload data first.")
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
            "SVM": SVC(probability=True)
        }

        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)

            results.append({
                "Model": name,
                "Train": train_acc,
                "Validation": val_acc
            })

        best_index = np.argmax([r["Validation"] for r in results])

        for i, (name, model) in enumerate(models.items()):
            st.subheader(name)

            if i == best_index:
                st.success("⭐ BEST MODEL")

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            col1, col2 = st.columns(2)

            with col1:
                st.write("Train:", round(results[i]["Train"], 4))
                plot_conf_matrix(confusion_matrix(y_train, train_pred), "Train")

            with col2:
                st.write("Validation:", round(results[i]["Validation"], 4))
                plot_conf_matrix(confusion_matrix(y_val, val_pred), "Validation")

        st.session_state["models"] = models
        st.session_state["results"] = results
        st.session_state["best_index"] = best_index

# =========================
# TESTING TAB
# =========================
with tab3:
    if "models" not in st.session_state:
        st.warning("Train models first.")
    else:
        test_files = st.file_uploader("Upload Testing Dataset", accept_multiple_files=True)

        if test_files:
            X_test, y_test = build_dataset(test_files)

            fig, ax = plt.subplots()
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
            st.pyplot(fig)

            for i, (name, model) in enumerate(st.session_state["models"].items()):
                st.subheader(name)

                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)

                if i == st.session_state["best_index"]:
                    st.success("⭐ BEST MODEL")

                st.write("Test Accuracy:", round(acc, 4))
                plot_conf_matrix(confusion_matrix(y_test, pred), "Test")

            # --- PER FILE ANALYSIS ---
            st.header("📁 Per-File Analysis (Best Model)")

            best_model = list(st.session_state["models"].values())[st.session_state["best_index"]]

            for file in test_files:
                with st.expander(file.name):

                    result = analyze_file(file, best_model)

                    if result is None:
                        st.warning("No hits detected.")
                        continue

                    signal, sr, hits, boundaries, preds, confidence = result

                    is_bad = confidence > 0.5

                    if is_bad:
                        st.error(f"DEFECT DETECTED ({confidence*100:.1f}%)")
                    else:
                        st.success(f"HEALTHY ({(1-confidence)*100:.1f}%)")

                    good_hits = np.sum(preds == 0)
                    bad_hits = np.sum(preds == 1)

                    st.write(f"GOOD: {good_hits} | BAD: {bad_hits}")

                    for i, p in enumerate(preds):
                        st.write(f"Hit {i+1}: {'BAD' if p==1 else 'GOOD'}")

                    fig, ax = plt.subplots(figsize=(7, 2))
                    ax.plot(signal, alpha=0.4)
                    for s, e in boundaries:
                        ax.axvspan(s, e, alpha=0.3)
                    st.pyplot(fig)
