"""Colab/Kaggle-friendly Craftly scratch GPU smoke entrypoint.

Usage:
  python deploy/gpu/run_scratch_gpu_smoke.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.craftly.model_ops.gpu_smoke import run_scratch_gpu_smoke  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Craftly scratch decoder GPU smoke.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="artifacts/craftly/gpu-smoke")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_scratch_gpu_smoke(
        device=args.device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        steps=args.steps,
        dtype=args.dtype,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

