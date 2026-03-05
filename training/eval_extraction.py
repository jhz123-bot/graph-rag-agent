"""Evaluate extraction quality metrics from prediction JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="jsonl with text/entities/relations")
    args = parser.parse_args()

    path = Path(args.pred)
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    json_fail = 0
    span_fail = 0
    dup_fail = 0
    rel_ok = 0
    rel_total = 0
    graph_ok = 0

    for row in rows:
        if not isinstance(row, dict):
            json_fail += 1
            continue
        text = row.get("text", "")
        entities = row.get("entities", [])
        relations = row.get("relations", [])

        seen = set()
        for e in entities:
            key = (e.get("name"), e.get("type"))
            if key in seen:
                dup_fail += 1
            seen.add(key)
            span = e.get("span", "")
            if span and span not in text:
                span_fail += 1

        ids = {e.get("id") for e in entities}
        ok_in_row = 0
        for r in relations:
            rel_total += 1
            if r.get("source") in ids and r.get("target") in ids:
                rel_ok += 1
                ok_in_row += 1
        if ok_in_row > 0:
            graph_ok += 1

    total = max(len(rows), 1)
    print(f"JSON失败率: {json_fail/total:.2%}")
    print(f"span无法定位率: {span_fail/max(sum(len(r.get('entities', [])) for r in rows),1):.2%}")
    print(f"重复实体率: {dup_fail/max(sum(len(r.get('entities', [])) for r in rows),1):.2%}")
    print(f"关系合法率: {rel_ok/max(rel_total,1):.2%}")
    print(f"Graph查询成功率: {graph_ok/total:.2%}")


if __name__ == "__main__":
    main()
