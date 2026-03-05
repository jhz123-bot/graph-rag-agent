import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGAL_DOMAIN_DIR = Path(
    os.getenv("LEGAL_DOMAIN_DIR", PROJECT_ROOT / "domains" / "legal")
).expanduser()
LOCAL_EXTRACTOR_MODEL_NAME = os.getenv(
    "LOCAL_EXTRACTOR_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"
)
LOCAL_EXTRACTOR_DEVICE = os.getenv("LOCAL_EXTRACTOR_DEVICE", "cpu")
LOCAL_EXTRACTOR_MAX_NEW_TOKENS = int(os.getenv("LOCAL_EXTRACTOR_MAX_NEW_TOKENS", "512"))
LOCAL_EXTRACTOR_USE_MOCK = os.getenv("LOCAL_EXTRACTOR_USE_MOCK", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
}


class LocalExtractor:
    """Small-model extractor for entity/relation extraction.

    Notes:
        - CPU demo friendly: default uses mock mode.
        - Real model path uses HuggingFace transformers causal LM.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        use_mock: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name or LOCAL_EXTRACTOR_MODEL_NAME
        self.device = device or LOCAL_EXTRACTOR_DEVICE
        self.max_new_tokens = max_new_tokens or LOCAL_EXTRACTOR_MAX_NEW_TOKENS
        self.use_mock = LOCAL_EXTRACTOR_USE_MOCK if use_mock is None else use_mock
        self._tokenizer = None
        self._model = None
        self._prompt_template = self._load_prompt_template()

        if not self.use_mock:
            self._load_transformer_model()

    def _load_prompt_template(self) -> str:
        prompt_file = Path(LEGAL_DOMAIN_DIR) / "prompts" / "entity_extract.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        return "Extract entities and relations as strict JSON."

    def _load_transformer_model(self) -> None:
        # 延迟导入，保证无 transformers 环境也能运行 mock demo
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self._model.to(self.device)

    def _mock_extract(self, text: str) -> Dict[str, Any]:
        # 基于规则的 mock 结果，保证 CPU demo 可跑
        entities = [
            {"id": "e1", "name": "案件A", "type": "Case", "span": "案件", "evidence": "案件"},
            {"id": "e2", "name": "被告人张三", "type": "Party", "span": "张三", "evidence": "张三"},
            {"id": "e3", "name": "盗窃行为", "type": "Act", "span": "盗窃", "evidence": "盗窃"},
            {"id": "e4", "name": "非法占有目的", "type": "MentalState", "span": "非法占有", "evidence": "非法占有"},
            {"id": "e5", "name": "被害人陈述", "type": "Evidence", "span": "被害人陈述", "evidence": "被害人陈述"},
            {"id": "e6", "name": "盗窃罪", "type": "Charge", "span": "盗窃罪", "evidence": "盗窃罪"},
            {"id": "e7", "name": "刑法第二百六十四条", "type": "LawArticle", "span": "第二百六十四条", "evidence": "第二百六十四条"},
        ]
        relations = [
            {"source": "e1", "target": "e3", "type": "HAS_ACT", "evidence": "实施盗窃"},
            {"source": "e3", "target": "e4", "type": "HAS_MENTAL_STATE", "evidence": "非法占有目的"},
            {"source": "e3", "target": "e6", "type": "MATCHES_ELEMENT", "evidence": "构成盗窃罪"},
            {"source": "e6", "target": "e7", "type": "APPLIES_ARTICLE", "evidence": "适用刑法第二百六十四条"},
            {"source": "e3", "target": "e5", "type": "SUPPORTED_BY", "evidence": "有被害人陈述"},
        ]
        if "受伤" in text:
            entities.append({"id": "e8", "name": "轻伤结果", "type": "Result", "span": "受伤", "evidence": "受伤"})
            relations.append({"source": "e3", "target": "e8", "type": "LEADS_TO", "evidence": "导致受伤"})
        return {"entities": entities, "relations": relations}

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract structured entities and relations from raw text."""
        if self.use_mock:
            return self._mock_extract(text)

        prompt = f"{self._prompt_template}\n\n输入文本:\n{text}\n\n输出:"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        json_str = generated.split("输出:", maxsplit=1)[-1].strip()
        return json.loads(json_str)
