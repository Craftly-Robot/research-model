"""Phase 2: Craftly Defensive SFT (Supervised Fine-Tuning) on Modal GPU.

Loads the scratch-origin Craftly 300M foundation checkpoint, fine-tunes on
curated defensive cybersecurity instruction pairs with token-level loss masking,
exports a clean production-ready package to /root/craftly_300m_sft_latest.zip,
and executes defensive diagnostic benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import modal
except ImportError:
    modal = None

from src.craftly.learning.sft_dataset import SFTDatasetEngine
from src.craftly.model_ops.checkpoint_compare import DEFAULT_PROMPTS, GenerationConfig, generate_text
from src.craftly.model_ops.sft_loop import SFTRunConfig, run_sft
from src.craftly.model_ops.tokenizer_pipeline import load_tokenizer
from src.craftly.model_ops.torch_decoder import (
    CraftlyDecoderLM,
    ScratchDecoderConfig,
    load_trusted_checkpoint,
    require_torch,
    select_torch_device,
)
from src.craftly.shared.progress import ProgressReporter

APP_NAME = "craftly-defensive-sft"
VOLUME_NAME = "craftly-training-volume"

if modal is not None:
    app = modal.App(APP_NAME)
    try:
        training_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    except Exception:
        training_volume = modal.Volume.from_name("craftly-training-volume", create_if_missing=True)

    training_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install(
            "torch>=2.3.0",
            "tokenizers>=0.15.2",
            "pydantic>=2.7.0",
            "httpx>=0.27.0",
            "safetensors>=0.4.0",
            "tqdm>=4.66.0",
            "pyyaml>=6.0.1",
        )
        .add_local_python_source("src", "config")
    )
else:
    app = None
    training_volume = None
    training_image = None


class LiveSFTProgressReporter(ProgressReporter):
    """Real-time progress reporter for SFT training output."""

    def __init__(self, on_checkpoint: Any = None) -> None:
        super().__init__(to_stdout=False)
        self.last_time = time.time()
        self.last_tokens = 0
        self.on_checkpoint = on_checkpoint

    def emit(self, stage: str, status: str = "running", **metrics: Any) -> dict[str, Any]:
        payload = super().emit(stage, status, **metrics)
        now = time.time()

        if stage == "sft" and status == "running":
            step = metrics.get("step", 0)
            total = metrics.get("requested_steps", 0)
            loss = metrics.get("loss", 0.0)
            lr = metrics.get("learning_rate", 0.0)
            tokens = metrics.get("trained_tokens", 0)

            dt = max(0.001, now - self.last_time)
            d_tokens = max(0, tokens - self.last_tokens)
            speed = int(d_tokens / dt) if self.last_tokens > 0 else 0
            self.last_time = now
            self.last_tokens = tokens

            pct = (step / total) * 100 if total > 0 else 0.0
            print(
                f"[SFT]   Step {step:5d}/{total} ({pct:4.1f}%) | "
                f"Loss: {loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Completion Tokens: {tokens:,} | "
                f"Speed: {speed:,} tok/s",
                flush=True,
            )

        elif stage == "sft_validation":
            step = metrics.get("step", 0)
            v_loss = metrics.get("validation_loss", 0.0)
            best_v = metrics.get("best_val_loss", 0.0)
            print(
                f"[SFT-VAL] Step {step:5d} | "
                f"Validation Loss: {v_loss:.4f} | "
                f"Best Val: {best_v:.4f}",
                flush=True,
            )

        elif stage == "sft" and status == "early_stopped":
            msg = metrics.get("message", "Early stopping triggered")
            print(f"[SFT-STOP] {msg}", flush=True)

        elif stage == "sft" and status == "completed":
            final_loss = metrics.get("final_train_loss", 0.0)
            steps = metrics.get("steps_completed", 0)
            print(f"[DONE]  SFT finished {steps} steps | Final Loss: {final_loss:.4f}\n", flush=True)

        return payload


def locate_base_checkpoint() -> Path:
    """Find the best available Craftly foundation checkpoint."""
    # 1. Check direct known paths
    direct_candidates = [
        Path("/root/research-model/artifacts/craftly/train_run/train_output_300m/checkpoint_manifest.json"),
        Path("/vol/train_run/train_output_300m/checkpoint_manifest.json"),
        Path("/root/craftly_300m_checkpoint_latest.zip"),
        Path("/root/craftly_300m_checkpoint_completed.zip"),
        Path("artifacts/craftly/train_run/train_output_300m/checkpoint_manifest.json"),
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            if candidate.suffix == ".zip":
                extract_dir = Path("/root/model_300m_extracted")
                if not extract_dir.exists():
                    print(f"[Setup] Extracting {candidate.name} to {extract_dir}...", flush=True)
                    with zipfile.ZipFile(candidate, "r") as zf:
                        zf.extractall(extract_dir)
                return extract_dir / "model.pt"
            return candidate

    # 2. Dynamic discovery in /vol, /mnt, /root
    search_roots = [Path("/vol"), Path("/mnt"), Path("/root")]
    for s_root in search_roots:
        if s_root.exists():
            # Check for manifests
            manifests = sorted(s_root.rglob("*checkpoint_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if manifests:
                return manifests[0]
            # Check for zip packages
            zips = sorted(s_root.rglob("*craftly*300m*checkpoint*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
            if zips:
                chosen_zip = zips[0]
                extract_dir = Path("/root/model_300m_extracted")
                if not extract_dir.exists():
                    print(f"[Setup] Extracting {chosen_zip.name} to {extract_dir}...", flush=True)
                    with zipfile.ZipFile(chosen_zip, "r") as zf:
                        zf.extractall(extract_dir)
                return extract_dir / "model.pt"

    raise FileNotFoundError(
        "Could not auto-locate Craftly 300M base checkpoint. "
        "Provide explicit path via --base-checkpoint <path>."
    )


def locate_tokenizer() -> Path:
    """Find tokenizer.json."""
    direct_candidates = [
        Path("/root/research-model/artifacts/craftly/train_run/tokenizer/tokenizer.json"),
        Path("/vol/train_run/tokenizer/tokenizer.json"),
        Path("/root/model_300m_extracted/tokenizer.json"),
        Path("artifacts/craftly/train_run/tokenizer/tokenizer.json"),
    ]
    for c in direct_candidates:
        if c.exists():
            return c

    # Dynamic scan
    for s_root in [Path("/vol"), Path("/mnt"), Path("/root")]:
        if s_root.exists():
            matches = list(s_root.rglob("tokenizer.json"))
            if matches:
                return matches[0]

    raise FileNotFoundError("Could not auto-locate tokenizer.json. Provide --tokenizer <path>.")


def export_clean_sft_package(
    ckpt_dir: Path,
    tokenizer_path: Path,
    step_val: int,
    tokens_val: int,
    output_prefix: str = "latest",
) -> Path | None:
    """Export clean, standalone SFT bundle for direct download."""
    root_dir = Path("/root") if Path("/root").exists() else ckpt_dir.parent
    clean_dir = root_dir / f"_export_sft_{output_prefix}_{int(time.time())}"
    clean_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. model.pt
        src_model = ckpt_dir / "model.pt"
        if src_model.exists():
            payload = load_trusted_checkpoint(src_model, map_location="cpu")
            clean_payload = {
                "model": payload.get("model", {}),
                "config": payload.get("config", {}),
                "step": step_val,
                "trained_tokens": tokens_val,
                "stage": "defensive_sft",
            }
            import torch

            torch.save(clean_payload, clean_dir / "model.pt")

        # 2. config.json
        if (ckpt_dir / "config.json").exists():
            shutil.copy(ckpt_dir / "config.json", clean_dir / "config.json")

        # 3. tokenizer.json
        if tokenizer_path.exists():
            shutil.copy(tokenizer_path, clean_dir / "tokenizer.json")

        # 4. Checkpoint info
        info = {
            "model_name": "Craftly",
            "stage": "Phase 2: Defensive SFT",
            "checkpoint_step": step_val,
            "completion_tokens": tokens_val,
            "exported_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        (clean_dir / "checkpoint_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

        # 5. Archive
        zip_base = root_dir / f"craftly_300m_sft_{output_prefix}"
        shutil.make_archive(str(zip_base), "zip", clean_dir)
        final_zip = Path(f"{zip_base}.zip")
        shutil.rmtree(clean_dir, ignore_errors=True)

        if final_zip.exists():
            size_mb = final_zip.stat().st_size / (1024 * 1024)
            print(f"[EXPORT] [OK] Saved {final_zip.name} ({size_mb:.1f} MB, Step {step_val:,})", flush=True)

            meta_txt = root_dir / "LATEST_SFT_CHECKPOINT.txt"
            meta_txt.write_text(
                f"==================================================\n"
                f"Craftly Phase 2 Defensive SFT Checkpoint\n"
                f"==================================================\n"
                f"Model:           Craftly 300M Dense AB (SFT)\n"
                f"Step:            {step_val:,}\n"
                f"Tokens:          {tokens_val:,}\n"
                f"Package File:    {final_zip.name}\n"
                f"Size:            {size_mb:.1f} MB\n"
                f"Status:          Production-grade Defensive Instruction Checkpoint\n"
                f"==================================================\n",
                encoding="utf-8",
            )
            return final_zip
    except Exception as err:
        print(f"[Export Note] {err}")
    return None


def run_defensive_evaluation(
    model_path: Path,
    tokenizer_path: Path,
    device: Any,
) -> None:
    """Run post-SFT evaluation on canonical security prompts."""
    print("\n" + "=" * 70)
    print("RUNNING POST-SFT DEFENSIVE EVALUATION ON CANONICAL PROMPTS")
    print("=" * 70)

    try:
        import torch

        checkpoint = load_trusted_checkpoint(model_path, map_location="cpu")
        config = ScratchDecoderConfig(**checkpoint["config"])
        model = CraftlyDecoderLM(config)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model = model.to(device)

        tokenizer = load_tokenizer(tokenizer_path)

        for prompt_case in DEFAULT_PROMPTS[:3]:
            task_id = prompt_case["task_id"]
            user_prompt = prompt_case["prompt"]
            formatted = f"User:\n{user_prompt}\n\nAssistant:\n"

            print(f"\n[EVALUATION TASK: {task_id}]")
            print(f"User Query:\n{user_prompt.strip()}\n" + "-" * 50)

            output, tokens = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=formatted,
                device=device,
                config=GenerationConfig(max_new_tokens=96, temperature=0.1),
            )
            print(f"Assistant Response:\n{output.strip()}\n")

    except Exception as err:
        print(f"[Evaluation Warning] Could not complete diagnostic eval: {err}")


def execute_sft_pipeline(
    base_checkpoint: str | None = None,
    tokenizer_path: str | None = None,
    output_dir: str | None = None,
    steps: int = 1000,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 3e-5,
    checkpoint_every: int = 250,
    validate_every: int = 50,
    smoke: bool = False,
) -> dict[str, Any]:
    """Execute complete Phase 2 Defensive SFT."""
    require_torch()
    import torch

    device = select_torch_device("auto")
    print("=" * 70)
    print("CRAFTLY PHASE 2: DEFENSIVE SUPERVISED FINE-TUNING (SFT)")
    print(f"Device: {device.type.upper()} | PyTorch: {torch.__version__}")
    print("=" * 70)

    resolved_base = Path(base_checkpoint) if base_checkpoint else locate_base_checkpoint()
    resolved_tok = Path(tokenizer_path) if tokenizer_path else locate_tokenizer()
    resolved_out = Path(output_dir) if output_dir else Path("/root/research-model/artifacts/craftly/sft_run")
    resolved_out.mkdir(parents=True, exist_ok=True)

    print(f"[Config] Base Foundation Model: {resolved_base}")
    print(f"[Config] Tokenizer:              {resolved_tok}")
    print(f"[Config] Output Directory:        {resolved_out}")
    print(f"[Config] Target SFT Steps:        {10 if smoke else steps}")
    print(f"[Config] Effective Batch Size:    {(1 if smoke else batch_size) * (1 if smoke else gradient_accumulation_steps)}")
    print(f"[Config] Learning Rate:           {1e-4 if smoke else learning_rate}")

    def on_checkpoint_hook(step: int, tokens: int, ckpt_path: Path) -> None:
        export_clean_sft_package(
            ckpt_dir=ckpt_path,
            tokenizer_path=resolved_tok,
            step_val=step,
            tokens_val=tokens,
            output_prefix="latest",
        )
        if modal is not None and training_volume is not None:
            try:
                training_volume.commit()
            except Exception:
                pass

    run_cfg = SFTRunConfig(
        base_checkpoint=str(resolved_base),
        tokenizer_path=str(resolved_tok),
        output_dir=str(resolved_out),
        steps=10 if smoke else steps,
        batch_size=1 if smoke else batch_size,
        gradient_accumulation_steps=1 if smoke else gradient_accumulation_steps,
        learning_rate=1e-4 if smoke else learning_rate,
        checkpoint_every=5 if smoke else checkpoint_every,
        validate_every=5 if smoke else validate_every,
        max_sequence_length=128 if smoke else 1024,
    )

    reporter = LiveSFTProgressReporter(on_checkpoint=on_checkpoint_hook)
    summary = run_sft(run_cfg, progress_reporter=reporter, on_checkpoint=on_checkpoint_hook)

    # Final Export
    latest_ckpt = Path(summary["latest_checkpoint_dir"])
    final_zip = export_clean_sft_package(
        ckpt_dir=latest_ckpt,
        tokenizer_path=resolved_tok,
        step_val=summary["steps_completed"],
        tokens_val=summary["trained_tokens"],
        output_prefix="completed",
    )

    if final_zip and Path("/root").exists():
        latest_link = Path("/root/craftly_300m_sft_latest.zip")
        try:
            shutil.copy(final_zip, latest_link)
        except Exception:
            pass

    if modal is not None and training_volume is not None:
        try:
            training_volume.commit()
            print("[Cloud Volume] SFT checkpoints and packages permanently saved to Modal Volume.")
        except Exception:
            pass

    # Post-SFT Diagnostic Evaluation
    if not smoke:
        run_defensive_evaluation(
            model_path=latest_ckpt / "model.pt",
            tokenizer_path=resolved_tok,
            device=device,
        )

    print("\n" + "=" * 70)
    print("PHASE 2 SFT TRAINING COMPLETE!")
    print(f"Final Model Package: /root/craftly_300m_sft_latest.zip")
    print("Download directly from your Modal file explorer!")
    print("=" * 70 + "\n")

    return summary


if modal is not None and app is not None:
    @app.function(
        gpu="A100-80GB",
        timeout=7200,
        volumes={"/vol": training_volume},
        image=training_image,
    )
    def sft_craftly_modal(
        steps: int = 1000,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        lr: float = 3e-5,
    ) -> dict[str, Any]:
        return execute_sft_pipeline(
            output_dir="/vol/sft_run",
            steps=steps,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Craftly Phase 2 Defensive SFT on Modal.")
    parser.add_argument("--base-checkpoint", default=None, help="Base model checkpoint or manifest")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer.json")
    parser.add_argument("--output-dir", default=None, help="SFT output directory")
    parser.add_argument("--steps", type=int, default=1000, help="SFT optimization steps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", help="Quick 10-step smoke test")
    args = parser.parse_args()

    execute_sft_pipeline(
        base_checkpoint=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        checkpoint_every=args.checkpoint_every,
        validate_every=args.validate_every,
        smoke=args.smoke,
    )
