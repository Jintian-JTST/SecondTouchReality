# run_model.py
"""
运行程序：
- 从 text_model.pkl 加载已经训练好的模型参数
- 读取用户输入的一句描述
- 输出每个标签的概率（相似度表），从高到低排序
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import joblib
from sklearn.feature_extraction.text import HashingVectorizer

MODEL_PATH = Path("text_model.pkl")


def init_vectorizer(n_features: int) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
    )


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH.resolve()}，先运行 train_model.py 训练一下。")

    data = joblib.load(MODEL_PATH)
    clf = data["classifier"]
    label_encoder = data["label_encoder"]
    n_features = data.get("n_features", 2 ** 18)

    vectorizer = init_vectorizer(n_features)
    return clf, label_encoder, vectorizer


def infer_once(query: str, clf, label_encoder, vectorizer, top_k: int = 5) -> List[Dict[str, Any]]:
    Xq = vectorizer.transform([query])

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(Xq)[0]
    else:
        # 理论上 loss="log_loss" 会有 predict_proba，这只是兜底
        scores = clf.decision_function(Xq)[0]
        scores = np.atleast_1d(scores)
        e = np.exp(scores - np.max(scores))
        probs = e / e.sum()

    indices = np.argsort(-probs)  # 从大到小排
    results: List[Dict[str, Any]] = []
    for idx in indices[:top_k]:
        label = label_encoder.inverse_transform([idx])[0]
        results.append(
            {
                "label": label,
                "prob": float(probs[idx]),
            }
        )
    return results


def main():
    print("=== 文本 -> 物体标签 推理程序 ===")
    clf, label_encoder, vectorizer = load_model()
    print(f"已加载模型: {MODEL_PATH.resolve()}")
    print(f"支持的标签: {list(label_encoder.classes_)}")
    print("输入一句描述，我会输出最可能的几个标签及其概率。")
    print("输入空行 或 exit / quit 退出。\n")

    while True:
        try:
            q = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not q or q.lower() in {"exit", "quit"}:
            print("退出。")
            break

        results = infer_once(q, clf, label_encoder, vectorizer, top_k=5)
        print(f"\n结果（从高到低）：")
        for r in results:
            print(f"  label={r['label']!r}  prob={r['prob']:.3f}")
        print()

    print("再见。")


if __name__ == "__main__":
    main()
