"""Contrastive embedding fine-tuning entry for legal term pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="training/legal_embedding_pairs.jsonl")
    args = parser.parse_args()

    path = Path(args.pairs)
    if not path.exists():
        print(f"pairs file not found: {path}")
        return

    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            _ = json.loads(line)
            count += 1

    print(f"loaded contrastive pairs: {count}")
    print("请使用 sentence-transformers Trainer 执行对比学习训练。")


if __name__ == "__main__":
    main()
