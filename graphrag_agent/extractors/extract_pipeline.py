import json
import os
import re
from typing import Any, Dict, Optional

from graphrag_agent.extractors.local_extractor import LocalExtractor
from graphrag_agent.extractors.validator import ExtractionValidator, ValidationError

ENABLE_LOCAL_EXTRACTOR = os.getenv("ENABLE_LOCAL_EXTRACTOR", "false").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
ENABLE_EXTRACT_FALLBACK = os.getenv("ENABLE_EXTRACT_FALLBACK", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
}


class ExtractPipeline:
    """Extraction pipeline: local small model -> validator -> fallback LLM/rules."""

    def __init__(self, local_extractor: Optional[LocalExtractor] = None) -> None:
        self.enabled = ENABLE_LOCAL_EXTRACTOR
        self.local_extractor = local_extractor or LocalExtractor()
        self.validator = ExtractionValidator()

    def _fallback_extract(self, text: str) -> Dict[str, Any]:
        # 轻量 fallback：优先规则，避免 demo 依赖外部 LLM。
        names = []
        for pattern in [r"被告人([\u4e00-\u9fa5]{1,4})", r"嫌疑人([\u4e00-\u9fa5]{1,4})"]:
            names.extend(re.findall(pattern, text))
        party_name = names[0] if names else "当事人"

        entities = [
            {"id": "c1", "name": "本案", "type": "Case", "span": "案件", "evidence": "案件" if "案件" in text else ""},
            {"id": "p1", "name": party_name, "type": "Party", "span": party_name if party_name in text else "", "evidence": party_name if party_name in text else ""},
            {"id": "a1", "name": "盗窃行为", "type": "Act", "span": "盗窃" if "盗窃" in text else "", "evidence": "盗窃" if "盗窃" in text else ""},
            {"id": "m1", "name": "非法占有目的", "type": "MentalState", "span": "非法占有" if "非法占有" in text else "", "evidence": "非法占有" if "非法占有" in text else ""},
            {"id": "ch1", "name": "盗窃罪", "type": "Charge", "span": "盗窃罪" if "盗窃罪" in text else "", "evidence": "盗窃罪" if "盗窃罪" in text else ""},
            {"id": "law1", "name": "刑法第二百六十四条", "type": "LawArticle", "span": "第二百六十四条" if "第二百六十四条" in text else "", "evidence": "第二百六十四条" if "第二百六十四条" in text else ""},
            {"id": "evi1", "name": "被害人陈述", "type": "Evidence", "span": "被害人陈述" if "被害人陈述" in text else "", "evidence": "被害人陈述" if "被害人陈述" in text else ""},
        ]
        entities = [e for e in entities if e.get("span") or e["type"] in {"Case", "Act", "Charge"}]

        relations = [
            {"source": "c1", "target": "a1", "type": "HAS_ACT", "evidence": "盗窃" if "盗窃" in text else "案件"},
            {"source": "a1", "target": "m1", "type": "HAS_MENTAL_STATE", "evidence": "非法占有" if "非法占有" in text else "案件"},
            {"source": "a1", "target": "ch1", "type": "MATCHES_ELEMENT", "evidence": "盗窃罪" if "盗窃罪" in text else "案件"},
            {"source": "ch1", "target": "law1", "type": "APPLIES_ARTICLE", "evidence": "第二百六十四条" if "第二百六十四条" in text else "案件"},
            {"source": "a1", "target": "evi1", "type": "SUPPORTED_BY", "evidence": "被害人陈述" if "被害人陈述" in text else "案件"},
        ]
        return {"entities": entities, "relations": relations}

    def extract(self, text: str) -> Dict[str, Any]:
        if not self.enabled:
            result = self._fallback_extract(text)
            return self.validator.validate_or_raise(result, text)

        try:
            local_result = self.local_extractor.extract(text)
            return self.validator.validate_or_raise(local_result, text)
        except Exception:
            if not ENABLE_EXTRACT_FALLBACK:
                raise
            fallback_result = self._fallback_extract(text)
            return self.validator.validate_or_raise(fallback_result, text)
