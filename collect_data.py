# collect_data.py
"""
交互式采集数据：
- 输入一句描述
- 输入对应的物体标签/编号（随便写：'apple' / '101' 都行）
- 追加存到 text_object_dataset.jsonl 里（一行一个 json）
"""

import json
from pathlib import Path
from datetime import datetime

DATA_PATH = Path("text_object_dataset.jsonl")


def append_example(text: str, label: str) -> None:
    record = {
        "text": text,
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with DATA_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    print("=== Interactive Data Collection ===")
    print(f"Data will be appended to file: {DATA_PATH.resolve()}")
    print("Each entry = a description + an object label/ID")
    print("Enter an empty line or 'exit' / 'quit' to finish.\n")

    while True:
        try:
            text = input("Description text> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not text or text.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        label = input("Object label/ID> ").strip()
        if not label:
            print("Label is empty, skipping this entry.")
            continue

        append_example(text, label)
        print("✅ Saved.")

    print("\nAll data has been written to the database file.")

if __name__ == "__main__":
    main()
