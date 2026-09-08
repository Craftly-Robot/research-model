"""Entrypoint for Craftly Catastrophic Forgetting and Knowledge Retention evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.craftly.learning.continual import (
    CANONICAL_FORGETTING_SUITE,
    ForgettingReport,
    evaluate_continual_perplexity,
)
from src.craftly.model_ops.foundation import ScratchDecoderConfig
from src.craftly.model_ops.sft_loop import resolve_base_model_assets
from src.craftly.model_ops.tokenizer_pipeline import load_tokenizer
from src.craftly.model_ops.torch_decoder import (
    CraftlyDecoderLM,
    require_torch,
    select_torch_device,
)


def load_model_from_checkpoint(checkpoint_path: Path, device: str) -> tuple[CraftlyDecoderLM, Any]:
    require_torch()
    _, state_dict, config_dict = resolve_base_model_assets(checkpoint_path)
    arch_config = ScratchDecoderConfig(**config_dict)
    model = CraftlyDecoderLM(arch_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, arch_config


def run_forgetting_evaluation(
    base_checkpoint: str,
    sft_checkpoint: str | None = None,
    continual_checkpoint: str | None = None,
    tokenizer_path: str = "config/tokenizer.json",
    output_path: str = "artifacts/craftly/forgetting_report.json",
    device: str = "auto",
) -> dict[str, Any]:
    dev = select_torch_device(device)
    tok_file = Path(tokenizer_path).resolve()
    if not tok_file.exists():
        for cand in [Path("tokenizer.json"), Path("/root/tokenizer.json"), Path("/root/model_300m_extracted/tokenizer.json")]:
            if cand.exists():
                tok_file = cand.resolve()
                break

    if not tok_file.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    tokenizer = load_tokenizer(tok_file)

    reports: dict[str, ForgettingReport] = {}

    print("=" * 70)
    print("CRAFTLY CATASTROPHIC FORGETTING BENCHMARK")
    print(f"Device: {dev.type.upper()} | Evaluation Suite: {len(CANONICAL_FORGETTING_SUITE)} tasks")
    print("=" * 70)

    # 1. Base Model Evaluation
    base_p = Path(base_checkpoint).resolve()
    print(f"\n[1/3] Evaluating Base Foundation Model: {base_p.name}...")
    base_model, _ = load_model_from_checkpoint(base_p, str(dev))
    reports["base_model"] = evaluate_continual_perplexity(base_model, tokenizer, device=str(dev))
    del base_model

    # 2. SFT Model Evaluation (if provided)
    if sft_checkpoint:
        sft_p = Path(sft_checkpoint).resolve()
        print(f"\n[2/3] Evaluating Standard SFT Model: {sft_p.name}...")
        sft_model, _ = load_model_from_checkpoint(sft_p, str(dev))
        reports["sft_standard"] = evaluate_continual_perplexity(sft_model, tokenizer, device=str(dev))
        del sft_model

    # 3. Continual / Merged Model Evaluation (if provided)
    if continual_checkpoint:
        cont_p = Path(continual_checkpoint).resolve()
        print(f"\n[3/3] Evaluating Continual / Merged Model: {cont_p.name}...")
        cont_model, _ = load_model_from_checkpoint(cont_p, str(dev))
        reports["continual_model"] = evaluate_continual_perplexity(cont_model, tokenizer, device=str(dev))
        del cont_model

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model Stage':<20} | {'Gen Loss':<10} | {'Def Loss':<10} | {'Gen PPL':<10} | {'Retention %':<12}")
    print("-" * 70)
    for stage, rep in reports.items():
        print(f"{stage:<20} | {rep.general_coding_loss:<10.4f} | {rep.defensive_security_loss:<10.4f} | {rep.perplexity_general:<10.2f} | {rep.general_retention_score:<12.2f}%")
    print("=" * 70)

    # Compute Comparative Forgetting Ratio
    analysis = {}
    if "sft_standard" in reports and "base_model" in reports:
        base_rep = reports["base_model"]
        sft_rep = reports["sft_standard"]
        forgetting_ratio = (sft_rep.general_coding_loss - base_rep.general_coding_loss) / max(0.001, base_rep.general_coding_loss)
        analysis["standard_sft_forgetting_ratio"] = round(forgetting_ratio, 4)
        print(f"\nStandard SFT Forgetting Ratio: {forgetting_ratio:+.2%}")

    if "continual_model" in reports and "base_model" in reports:
        base_rep = reports["base_model"]
        cont_rep = reports["continual_model"]
        cont_forgetting = (cont_rep.general_coding_loss - base_rep.general_coding_loss) / max(0.001, base_rep.general_coding_loss)
        analysis["continual_forgetting_ratio"] = round(cont_forgetting, 4)
        print(f"Continual / Merged Forgetting Ratio: {cont_forgetting:+.2%}")

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reports": {k: v.model_dump() for k, v in reports.items()},
        "analysis": analysis,
    }
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[Report Saved] -> {out_file}\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Catastrophic Forgetting in Craftly Models.")
    parser.add_argument("--base", required=True, help="Path to base pretrained model checkpoint")
    parser.add_argument("--sft", default=None, help="Path to standard SFT model checkpoint")
    parser.add_argument("--continual", default=None, help="Path to continual learning or merged checkpoint")
    parser.add_argument("--tokenizer", default="config/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--output", default="artifacts/craftly/forgetting_report.json", help="Output path for JSON report")
    parser.add_argument("--device", default="auto", help="PyTorch device")
    args = parser.parse_args()

    run_forgetting_evaluation(
        base_checkpoint=args.base,
        sft_checkpoint=args.sft,
        continual_checkpoint=args.continual,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
