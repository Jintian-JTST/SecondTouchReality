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
        analyzer="char_wb",       
        ngram_range=(3, 6),      
    )



def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find model file: {MODEL_PATH.resolve()}, please run train_model.py first.")

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
        scores = clf.decision_function(Xq)[0]
        scores = np.atleast_1d(scores)
        e = np.exp(scores - np.max(scores))
        probs = e / e.sum()

    indices = np.argsort(-probs) 
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
    print("=== text -> object label inference program ===")
    clf, label_encoder, vectorizer = load_model()
    print(f"Loaded model: {MODEL_PATH.resolve()}")
    print(f"Supported labels: {list(label_encoder.classes_)}")
    print("Enter a description, and I will output the most likely labels and their probabilities.")
    print("Enter an empty line or 'exit' / 'quit' to exit.\n")

    while True:
        try:
            q = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q or q.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        results = infer_once(q, clf, label_encoder, vectorizer, top_k=5)
        print(f"\nResults (from high to low):")
        for r in results:
            print(f"  label={r['label']!r}  prob={r['prob']:.3f}")
        print()

    print("Goodbye.")


if __name__ == "__main__":
    main()
