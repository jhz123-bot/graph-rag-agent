"""SFT (QLoRA) skeleton script for local extraction model."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="training/dataset.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", default="training/checkpoints/sft")
    args = parser.parse_args()

    # 这里保留轻量入口，避免 CPU demo 环境强依赖训练库。
    print("[SFT] 准备训练参数")
    print(f"dataset={args.dataset}")
    print(f"model={args.model}")
    print(f"output={args.output}")
    print("如需真实训练，请安装 transformers/peft/trl/bitsandbytes 并补全 Trainer 逻辑。")


if __name__ == "__main__":
    main()
