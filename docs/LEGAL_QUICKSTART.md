# LEGAL QUICKSTART

## 1. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 下载/准备数据

- 法律文本：自行放入 `documents/` 或准备 `.txt/.jsonl`。
- embedding 对比样本：`training/legal_embedding_pairs.jsonl`。

## 3. 运行 demo

```bash
python scripts/demo_legal_graphrag.py
```

默认可在 CPU 环境运行（使用 mock/local fallback）。

## 4. 训练模型

```bash
python training/build_dataset.py --input your_legal_text.txt --output training/dataset.jsonl
python training/sft_train.py --dataset training/dataset.jsonl
python training/grpo_train.py --dataset training/dataset.jsonl --num-candidates 4
python training/embedding_train.py --pairs training/legal_embedding_pairs.jsonl
python training/eval_extraction.py --pred training/dataset.jsonl
```

> 说明：GPU 仅用于训练阶段；demo 与数据构建支持 CPU 运行。
