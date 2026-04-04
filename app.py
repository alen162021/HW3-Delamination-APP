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

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Delamination ML Lab", layout="wide", page_icon="🔊")

st.title("🔊 Delamination Detection & Machine Learning Lab")

st.caption("""
This project was partially sponsored by a Teaching Innovation Program (TIP) grant,
Smart Materials and Structures Laboratory (SMSL), and Artificial Intelligent Laboratory
for Monitoring and Inspection, University of Houston.
""")

# =========================
# FUNCTIONS
# =========================
def load_audio(file):
    suffix = "." + file.name.split(".")[-1].lower()

    if file.size == 0:
        raise ValueError(f"{file.name} is empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        signal, sr = sf.read(tmp_path)

        if isinstance(signal, np.ndarray) and signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        return signal, sr

    except Exception:
        try:
            signal, sr = librosa.load(tmp_path, sr=22050, mono=True)
            return signal, sr
        except Exception as e:
            raise ValueError(f"Could not load {file.name}: {str(e)}")


def split_hits(signal, sr):
    signal = signal / (np.max(np.abs(signal)) + 1e-9)

    frame_length, hop_length = int(0.02 * sr), int(0.01 * sr)

    energy = librosa.feature.rms(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

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
            hits, _ = split_hits(signal, sr)

            if "_b" in file.name.lower():
                label = 1
            else:
                label = 0

            for h in hits:
                X.append(extract_features(h, sr))
                y.append(label)

        except Exception as e:
            st.warning(f"Skipping {file.name}: {e}")
            continue

    if len(X) == 0:
        return np.array([]), np.array([])

    return np.array(X), np.array(y)


def plot_conf_matrix(cm, title):
    fig, ax = plt.subplots()
    im = ax.imshow(cm)

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
    except Exception as e:
        st.error(str(e))
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
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📂 Data", "🤖 Training", "🧪 Testing"])

# =========================
# TAB 1: DATA
# =========================
with tab1:
    train_files = st.file_uploader("Upload HW3 Dataset (_g/_b)", accept_multiple_files=True)

    if train_files:
        X, y = build_dataset(train_files)

        if len(X) == 0:
            st.error("No valid audio features extracted.")
        else:
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
            st.session_state["train_files"] = train_files

# =========================
# TAB 2: TRAINING
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

        st.header("Model Performance")

        for name, model in models.items():
            model.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, model.predict(X_train))
            val_acc = accuracy_score(y_val, model.predict(X_val))

            results.append({"Model": name, "Train": train_acc, "Validation": val_acc})

        best_index = np.argmax([r["Validation"] for r in results])

        for i, (name, model) in enumerate(models.items()):
            st.subheader(name)

            if i == best_index:
                st.success("⭐ BEST MODEL")

            col1, col2 = st.columns(2)

            with col1:
                st.write("Train Acc:", round(results[i]["Train"], 4))
                plot_conf_matrix(
                    confusion_matrix(y_train, model.predict(X_train)),
                    "Train CM"
                )

            with col2:
                st.write("Val Acc:", round(results[i]["Validation"], 4))
                plot_conf_matrix(
                    confusion_matrix(y_val, model.predict(X_val)),
                    "Validation CM"
                )

        st.session_state["models"] = models
        st.session_state["results"] = results
        st.session_state["best_index"] = best_index

# =========================
# TAB 3: TESTING
# =========================
with tab3:
    if "models" not in st.session_state:
        st.warning("Train models first.")
    else:
        test_files = st.file_uploader("Upload HW2 Dataset", accept_multiple_files=True)

        if test_files:
            X_test, y_test = build_dataset(test_files)

            if len(X_test) > 0:
                fig, ax = plt.subplots()
                ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
                st.pyplot(fig)

            for i, (name, model) in enumerate(st.session_state["models"].items()):
                st.subheader(name)

                if len(X_test) > 0:
                    pred = model.predict(X_test)
                    acc = accuracy_score(y_test, pred)

                    if i == st.session_state["best_index"]:
                        st.success("⭐ BEST MODEL")

                    st.write("Test Accuracy:", round(acc, 4))

                    plot_conf_matrix(
                        confusion_matrix(y_test, pred),
                        "Test CM"
                    )

                    drop = st.session_state["results"][i]["Validation"] - acc
                    st.write("Accuracy Drop:", round(drop, 4))

            # =========================
            # PER FILE ANALYSIS
            # =========================
            st.header("📁 Per-File Analysis (Best Model)")

            best_model = list(st.session_state["models"].values())[
                st.session_state["best_index"]
            ]

            for file in test_files:
                with st.expander(file.name):

                    result = analyze_file(file, best_model)

                    if result is None:
                        st.warning("No valid hits detected.")
                        continue

                    signal, sr, hits, boundaries, preds, confidence = result

                    if confidence > 0.5:
                        st.error(f"DEFECT DETECTED ({confidence*100:.1f}%)")
                    else:
                        st.success(f"HEALTHY ({(1-confidence)*100:.1f}%)")

                    st.write(f"GOOD: {np.sum(preds == 0)} | BAD: {np.sum(preds == 1)}")

                    for i, p in enumerate(preds):
                        st.write(f"Hit {i+1}: {'BAD' if p==1 else 'GOOD'}")

                    fig, ax = plt.subplots(figsize=(7, 2))
                    ax.plot(signal, alpha=0.4)

                    for s, e in boundaries:
                        ax.axvspan(s, e, alpha=0.3)

                    st.pyplot(fig)
