import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# --- FEATURE EXTRACTION (PSD + MFCC) ---
def get_feature_vector(hits, sr):
    feature_matrix = []
    for h in hits:
        # MFCC (13 coefficients)
        mfccs = np.mean(librosa.feature.mfcc(y=h, sr=sr, n_mfcc=13), axis=1)
        # PSD (Mean Power)
        psd = np.mean(np.abs(np.fft.fft(h))**2)
        # Combine into a single vector (14 features total)
        feature_matrix.append(np.hstack([mfccs, psd]))
    return np.array(feature_matrix)

# --- TRAINING FUNCTION ---
def run_benchmarks(X, y, X_test=None, y_test=None):
    # Task 2.3: 70/30 Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='rbf', probability=True)
    }
    
    report = []
    figures = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        
        # Accuracies
        acc_train = accuracy_score(y_train, clf.predict(X_train))
        acc_val = accuracy_score(y_val, clf.predict(X_val))
        
        # Part 3: Robustness (HW2 Test Set)
        acc_test = accuracy_score(y_test, clf.predict(X_test)) if X_test is not None else 0
        
        report.append({
            "Model": name,
            "Train Acc": acc_train,
            "Val Acc": acc_val,
            "Test (HW2) Acc": acc_test,
            "Accuracy Drop": acc_val - acc_test
        })

        # Confusion Matrices
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(confusion_matrix(y_val, clf.predict(X_val)), annot=True, fmt='d', ax=ax[0], cmap="Blues")
        ax[0].set_title(f"{name} - Validation")
        if X_test is not None:
            sns.heatmap(confusion_matrix(y_test, clf.predict(X_test)), annot=True, fmt='d', ax=ax[1], cmap="Reds")
            ax[1].set_title(f"{name} - HW2 Test")
        figures[name] = fig
            
    return pd.DataFrame(report), figures
