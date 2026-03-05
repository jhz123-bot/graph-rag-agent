"""GRPO training entry (TRL GRPOTrainer compatible placeholder)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reward_functions import (
    dedup_reward,
    evidence_reward,
    graph_utility_reward,
    schema_reward,
    type_consistency_reward,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="training/dataset.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-candidates", type=int, default=4)
    args = parser.parse_args()

    print("[GRPO] 参数")
    print(f"dataset={args.dataset}, model={args.model}, K={args.num_candidates}")
    print("奖励函数已就绪:")
    print(schema_reward.__name__, type_consistency_reward.__name__, evidence_reward.__name__, dedup_reward.__name__, graph_utility_reward.__name__)
    print("如需真实 GRPO 训练，请安装 trl 并接入 GRPOTrainer。")


if __name__ == "__main__":
    main()
