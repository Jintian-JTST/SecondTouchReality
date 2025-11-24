# train_model.py
"""
训练程序：
- 从 text_object_dataset.jsonl 读取数据
- 把文本转成特征向量
- 用 SGDClassifier 训练一个多分类模型
- 模型 + 标签编码器 一起存到 text_model.pkl
- 如果已有模型，默认在上一次参数基础上继续 partial_fit
"""

import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import joblib

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path("cleaned_text_object_dataset.jsonl")
MODEL_PATH = Path("text_model.pkl")

# HashingVectorizer 的特征维度（可以调，但三个文件要一致）
N_FEATURES = 2 ** 18  # 262144 维，够用了


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
        raise RuntimeError("Data file is empty, please run collect_data.py to collect some data.")

    return texts, labels


def init_vectorizer() -> HashingVectorizer:
    # HashingVectorizer 无需 fit，是固定的 hash 特征映射
    return HashingVectorizer(
        n_features=N_FEATURES,
        alternate_sign=False,  # 避免正负号翻转，直觉一点
        norm="l2",
    )


def train():
    print("=== Train Text Classification Model ===")
    texts, labels = load_dataset()
    print(f"Loaded {len(texts)} samples.")

    unique_labels = sorted(set(labels))
    print(f"Total {len(unique_labels)} unique labels: {unique_labels}")

    vectorizer = init_vectorizer()
    X = vectorizer.transform(texts)

    # 是否已有旧模型
    if MODEL_PATH.exists():
        print(f"\nDetected existing model file: {MODEL_PATH.resolve()}")
        data = joblib.load(MODEL_PATH)
        clf: SGDClassifier = data["classifier"]
        label_encoder: LabelEncoder = data["label_encoder"]
        old_classes = set(label_encoder.classes_)

        new_label_set = set(unique_labels)
        if new_label_set.issubset(old_classes):
            # 没有新标签：在旧参数基础上继续训练
            print("No new labels detected, continuing training on existing model.")
            y = label_encoder.transform(labels)
            clf.partial_fit(X, y)
        else:
            # 有新标签：从头重训
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
        # 没有旧模型，第一次训练
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

    # 保存模型参数
    model_data = {
        "classifier": clf,
        "label_encoder": label_encoder,
        "n_features": N_FEATURES,
    }
    joblib.dump(model_data, MODEL_PATH)

    print("\n✅ Training complete.")
    print(f"Model saved to: {MODEL_PATH.resolve()}")
    print(f"Label classes: {list(label_encoder.classes_)}")


if __name__ == "__main__":
    train()
