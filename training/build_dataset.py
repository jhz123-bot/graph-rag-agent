"""Build legal extraction dataset JSONL.

Pipeline:
legal text -> LLM/mock extraction -> rule filter -> dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graphrag_agent.extractors.extract_pipeline import ExtractPipeline


def iter_texts(input_path: Path) -> list[str]:
    if input_path.suffix.lower() == ".jsonl":
        items = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line).get("text", ""))
        return items
    return [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="training/dataset.jsonl")
    args = parser.parse_args()

    texts = iter_texts(Path(args.input))
    pipeline = ExtractPipeline()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as wf:
        for text in texts:
            item = pipeline.extract(text)
            record = {"text": text, "entities": item["entities"], "relations": item["relations"]}
            wf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"dataset built: {out}")


if __name__ == "__main__":
    main()
