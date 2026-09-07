"""Kaggle/Colab entrypoint for Craftly scratch checkpoint comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.craftly.model_ops.checkpoint_compare import GenerationConfig, compare_checkpoints  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Craftly scratch checkpoints on a fixed coding/security prompt suite.")
    parser.add_argument("--training-report", help="pretrain_report.json or real_data_training_report.json")
    parser.add_argument("--baseline-manifest")
    parser.add_argument("--candidate-manifest")
    parser.add_argument("--tokenizer")
    parser.add_argument("--prompt-suite")
    parser.add_argument("--output-dir", default="artifacts/craftly/checkpoint-compare")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--stop-token", action="append", dest="stop_tokens")
    parser.add_argument("--max-repetition-ratio", type=float, default=0.72)
    return parser.parse_args()


def _load_training_payload(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    training = payload.get("training")
    if isinstance(training, dict):
        return {**payload, **training}
    return payload


def _resolve_args(args: argparse.Namespace) -> tuple[str, str, str]:
    baseline = args.baseline_manifest
    candidate = args.candidate_manifest
    tokenizer = args.tokenizer
    if args.training_report:
        payload = _load_training_payload(args.training_report)
        baseline = baseline or str(payload.get("checkpoint_manifest") or "")
        candidate = candidate or str(payload.get("best_checkpoint_manifest") or payload.get("checkpoint_manifest") or "")
        tokenizer = tokenizer or str(payload.get("tokenizer_path") or "")
    if not baseline or not candidate:
        raise ValueError("provide --training-report or both --baseline-manifest and --candidate-manifest")
    if not tokenizer:
        raise ValueError("provide --tokenizer or a training report containing tokenizer_path")
    return baseline, candidate, tokenizer


def main() -> None:
    args = parse_args()
    baseline, candidate, tokenizer = _resolve_args(args)
    report = compare_checkpoints(
        baseline_manifest=baseline,
        candidate_manifest=candidate,
        tokenizer_path=tokenizer,
        prompt_suite=args.prompt_suite,
        output_dir=args.output_dir,
        device=args.device,
        generation_config=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            stop_tokens=args.stop_tokens or GenerationConfig().stop_tokens,
            max_repetition_ratio=args.max_repetition_ratio,
        ),
    )
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
    raise SystemExit(0 if report.status not in {"regressed", "failed_generation_collapse", "failed_hallucination_guardrail"} else 1)


if __name__ == "__main__":
    main()

