
"""
Train program with evaluation:
- Reads clean_dataset JSONL (text,label)
- Uses HashingVectorizer + SGDClassifier (supports partial_fit)
- Saves model to text_model.pkl
- After training, computes training accuracy and prints classification report
"""

import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import joblib

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = Path("cleaned_text_object_dataset.jsonl")
MODEL_PATH = Path("text_model.pkl")
N_FEATURES = 2 ** 18  # 262144


def load_dataset() -> Tuple[List[str], List[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find data file: {DATA_PATH.resolve()}")

    texts: List[str] = []
    labels: List[str] = []

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            texts.append(rec["text"])
            labels.append(rec["label"])

    if not texts:
        raise RuntimeError("Data file is empty, please collect some data first.")

    return texts, labels


from sklearn.feature_extraction.text import HashingVectorizer

def init_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=N_FEATURES,
        alternate_sign=False,
        norm="l2",
        analyzer="char_wb",
        ngram_range=(3, 6),
    )



def train():
    print("=== Train Text Classification Model (with evaluation) ===")
    texts, labels = load_dataset()
    print(f"Loaded {len(texts)} samples.")

    unique_labels = sorted(set(labels))
    print(f"Total {len(unique_labels)} unique labels: {unique_labels}")

    vectorizer = init_vectorizer()
    X = vectorizer.transform(texts)

    if MODEL_PATH.exists():
        print(f"\nDetected existing model file: {MODEL_PATH.resolve()}")
        data = joblib.load(MODEL_PATH)
        clf: SGDClassifier = data["classifier"]
        label_encoder: LabelEncoder = data["label_encoder"]
        old_classes = set(label_encoder.classes_)

        new_label_set = set(unique_labels)
        if new_label_set.issubset(old_classes):
            print("No new labels detected, continuing training on existing model (partial_fit).")
            y = label_encoder.transform(labels)
            clf.partial_fit(X, y)
        else:
            print("⚠️ New labels detected, retraining model from scratch.")
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(labels)
            classes_indices = np.arange(len(label_encoder.classes_), dtype=np.int32)

            clf = SGDClassifier(
                loss="log_loss",
                max_iter=5,
                learning_rate="optimal",
                n_jobs=-1,
            )
            clf.partial_fit(X, y, classes=classes_indices)
    else:
        print("No old model found, training new model from scratch.")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        classes_indices = np.arange(len(label_encoder.classes_), dtype=np.int32)

        clf = SGDClassifier(
            loss="log_loss",
            max_iter=5,
            learning_rate="optimal",
            n_jobs=-1,
        )
        clf.partial_fit(X, y, classes=classes_indices)

    model_data = {
        "classifier": clf,
        "label_encoder": label_encoder,
        "n_features": N_FEATURES,
    }
    joblib.dump(model_data, MODEL_PATH, compress=3)

    print("\n Training complete.")
    print(f"Model saved to: {MODEL_PATH.resolve()}")
    print(f"Label classes: {list(label_encoder.classes_)}")

    try:
        if hasattr(clf, "predict"):
            y_pred_idx = clf.predict(X)
        else:
            scores = clf.decision_function(X)
            if scores.ndim == 1:
                y_pred_idx = (scores > 0).astype(int)
            else:
                y_pred_idx = scores.argmax(axis=1)
    except Exception as e:
        print("Could not run prediction on training set:", e)
        return

    inv_labels = label_encoder.inverse_transform(y_pred_idx)
    y_true_idx = label_encoder.transform(labels)
    y_true = label_encoder.inverse_transform(y_true_idx)

    acc = accuracy_score(y_true_idx, y_pred_idx)
    print(f"\nTraining accuracy: {acc:.4f}")

    print("\nClassification report (on training set):")
    print(classification_report(y_true, inv_labels, zero_division=0))


if __name__ == "__main__":
    train()
