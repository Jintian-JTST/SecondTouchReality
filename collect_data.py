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
    print("=== 文本-物体 数据采集器 ===")
    print(f"数据会被追加到文件: {DATA_PATH.resolve()}")
    print("每条数据 = 一句描述 + 一个物体标签/编号")
    print("输入空行 或 exit / quit 结束。\n")

    while True:
        try:
            text = input("描述 text> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not text or text.lower() in {"exit", "quit"}:
            print("结束采集。")
            break

        label = input("对应的物体标签/编号 label> ").strip()
        if not label:
            print("标签为空，跳过这一条。")
            continue

        append_example(text, label)
        print("✅ 已保存。")

    print("\n所有数据都已经写入数据库文件。")


if __name__ == "__main__":
    main()
