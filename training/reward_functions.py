"""Reward functions for GRPO extraction training."""

from __future__ import annotations

from typing import Any, Dict


def schema_reward(pred: Dict[str, Any]) -> float:
    return 1.0 if isinstance(pred, dict) and "entities" in pred and "relations" in pred else 0.0


def type_consistency_reward(pred: Dict[str, Any]) -> float:
    entities = {e.get("id"): e.get("type") for e in pred.get("entities", [])}
    score = 0.0
    for rel in pred.get("relations", []):
        if rel.get("source") in entities and rel.get("target") in entities and rel.get("type"):
            score += 1.0
    return score / max(len(pred.get("relations", [])), 1)


def evidence_reward(pred: Dict[str, Any], text: str) -> float:
    hits = 0
    total = 0
    for e in pred.get("entities", []):
        span = e.get("span", "")
        if span:
            total += 1
            if span in text:
                hits += 1
    for r in pred.get("relations", []):
        ev = r.get("evidence", "")
        if ev:
            total += 1
            if ev in text:
                hits += 1
    return hits / max(total, 1)


def dedup_reward(pred: Dict[str, Any]) -> float:
    seen = set()
    dup = 0
    for e in pred.get("entities", []):
        key = (e.get("name"), e.get("type"))
        if key in seen:
            dup += 1
        seen.add(key)
    return 1.0 - dup / max(len(pred.get("entities", [])), 1)


def graph_utility_reward(pred: Dict[str, Any]) -> float:
    entities = {e.get("id") for e in pred.get("entities", [])}
    ok = 0
    rels = pred.get("relations", [])
    for r in rels:
        if r.get("source") in entities and r.get("target") in entities:
            ok += 1
    return ok / max(len(rels), 1)
