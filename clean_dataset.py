# clean_dataset.py
"""
从原始 JSONL 数据里：
- 丢弃 text 中含有中文字符的样本；
- 只保留 "text" 和 "label" 字段（不再保留 timestamp）；
- 输出到一个新的 JSONL 文件。
"""

import json
import re
from pathlib import Path

# 输入 / 输出文件名，你可以自己改
INPUT_PATH = Path("text_object_dataset.jsonl")
OUTPUT_PATH = Path("cleaned_text_object_dataset.jsonl")

# 正则：匹配任意中文（CJK）字符
cn_pattern = re.compile(r"[\u4e00-\u9fff]")

def is_english_only(text: str) -> bool:
    """如果 text 里不含中文字符，就返回 True。"""
    return cn_pattern.search(text) is None

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"找不到输入文件: {INPUT_PATH.resolve()}")

    kept = 0
    dropped = 0

    with INPUT_PATH.open("r", encoding="utf-8") as fin, \
         OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            text = rec.get("text", "")
            label = rec.get("label", "")

            # 丢掉没有 text 或 label 的行
            if not text or label == "":
                dropped += 1
                continue

            # 丢掉含中文的样本
            if not is_english_only(text):
                dropped += 1
                continue

            clean_rec = {
                "text": text,
                "label": str(label),  # 统一成字符串
            }
            fout.write(json.dumps(clean_rec) + "\n")
            kept += 1

    print(f"done. kept={kept}, dropped={dropped}")
    print(f"clean file: {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
