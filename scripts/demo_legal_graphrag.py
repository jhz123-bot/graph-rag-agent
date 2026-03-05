"""Legal GraphRAG demo.

Run:
    python scripts/demo_legal_graphrag.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graphrag_agent.extractors import ExtractPipeline


DEMO_TEXT = (
    "本案件中，被告人张三以非法占有为目的，多次实施盗窃，窃得他人财物。"
    "经查，被害人陈述及监控视频能够相互印证。"
    "公诉机关指控其行为构成盗窃罪，建议适用刑法第二百六十四条。"
)


def build_paths(extraction: dict) -> list[dict]:
    rel_by_source = defaultdict(list)
    entities = {e["id"]: e for e in extraction["entities"]}
    for rel in extraction["relations"]:
        rel_by_source[rel["source"]].append(rel)

    paths = []
    for rel in extraction["relations"]:
        src = entities.get(rel["source"], {"name": rel["source"]})
        tgt = entities.get(rel["target"], {"name": rel["target"]})
        paths.append(
            {
                "path": f"{src['name']} -[{rel['type']}]-> {tgt['name']}",
                "evidence": rel.get("evidence", ""),
            }
        )
    return paths


def main() -> None:
    pipeline = ExtractPipeline()
    extraction = pipeline.extract(DEMO_TEXT)
    paths = build_paths(extraction)

    print("=== 抽取实体 ===")
    print(json.dumps(extraction["entities"], ensure_ascii=False, indent=2))

    print("\n=== 抽取关系 ===")
    print(json.dumps(extraction["relations"], ensure_ascii=False, indent=2))

    print("\n=== 图谱路径 + evidence ===")
    print(json.dumps(paths, ensure_ascii=False, indent=2))

    print("\n=== 验收统计 ===")
    print(f"实体数量: {len(extraction['entities'])}")
    print(f"关系数量: {len(extraction['relations'])}")
    all_rel_have_evidence = all(bool(r.get('evidence')) for r in extraction['relations'])
    print(f"关系证据齐全: {all_rel_have_evidence}")


if __name__ == "__main__":
    main()
