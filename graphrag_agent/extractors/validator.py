import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGAL_DOMAIN_DIR = Path(
    os.getenv("LEGAL_DOMAIN_DIR", PROJECT_ROOT / "domains" / "legal")
).expanduser()


class ValidationError(Exception):
    """Raised when extracted graph data is invalid."""


@dataclass
class ValidationReport:
    valid: bool
    errors: List[str]


class ExtractionValidator:
    """Validate extraction output against legal schema rules."""

    def __init__(self, schema_path: Path | None = None) -> None:
        self.schema_path = schema_path or (Path(LEGAL_DOMAIN_DIR) / "schema.yaml")
        self.schema = self._load_schema()
        self.entity_types: Set[str] = set(self.schema.get("entity_types", []))
        self.rel_types: Set[str] = set(self.schema.get("relationship_types", []))

    def _load_schema(self) -> Dict[str, Any]:
        if not self.schema_path.exists():
            return {"entity_types": [], "relationship_types": [], "constraints": {}}
        with self.schema_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _ensure_json(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, str):
            return json.loads(payload)
        if not isinstance(payload, dict):
            raise ValidationError("payload 必须是 dict 或 JSON 字符串")
        return payload

    def validate(self, payload: Any, source_text: str) -> ValidationReport:
        errors: List[str] = []
        try:
            data = self._ensure_json(payload)
        except Exception as exc:
            return ValidationReport(valid=False, errors=[f"JSON 非法: {exc}"])

        entities = data.get("entities", [])
        relations = data.get("relations", [])
        if not isinstance(entities, list) or not isinstance(relations, list):
            return ValidationReport(valid=False, errors=["entities/relations 必须为 list"])

        entity_ids = set()
        entity_names = set()
        for idx, entity in enumerate(entities):
            eid = entity.get("id")
            etype = entity.get("type")
            name = entity.get("name")
            span = entity.get("span", "")
            if not eid:
                errors.append(f"entity[{idx}] 缺少 id")
                continue
            if eid in entity_ids:
                errors.append(f"重复实体 id: {eid}")
            entity_ids.add(eid)
            if etype not in self.entity_types:
                errors.append(f"实体类型非法: {etype}")
            dedup_key = (name, etype)
            if dedup_key in entity_names:
                errors.append(f"重复实体 name/type: {name}/{etype}")
            entity_names.add(dedup_key)
            if span and span not in source_text:
                errors.append(f"span 无法定位: {span}")

        for ridx, rel in enumerate(relations):
            rtype = rel.get("type")
            src = rel.get("source")
            tgt = rel.get("target")
            if rtype not in self.rel_types:
                errors.append(f"关系类型非法: {rtype}")
            if src not in entity_ids or tgt not in entity_ids:
                errors.append(f"关系端点非法 relation[{ridx}]: {src}->{tgt}")
            evidence = rel.get("evidence", "")
            if evidence and evidence not in source_text:
                errors.append(f"relation evidence 无法定位: {evidence}")

        return ValidationReport(valid=(len(errors) == 0), errors=errors)

    def validate_or_raise(self, payload: Any, source_text: str) -> Dict[str, Any]:
        report = self.validate(payload, source_text)
        if not report.valid:
            raise ValidationError("; ".join(report.errors))
        return payload if isinstance(payload, dict) else json.loads(payload)
