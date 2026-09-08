"""Phase 3: Autonomous Defensive Agent & Sandbox Verification on Modal GPU.

Executes autonomous tool calling, unified diff patch synthesis, and sandbox
regression verification using the Phase 2 fine-tuned Craftly 300M model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import modal
except ImportError:
    modal = None

from src.craftly.learning.agentic_dataset import build_canonical_agentic_scenarios
from src.craftly.model_ops.native_backend import NativeCraftlyModelBackend
from src.craftly.runtime.agent_loop import AgenticRuntime, AgenticTask
from src.craftly.model_ops.torch_decoder import select_torch_device

APP_NAME = "craftly-agentic-execution"
VOLUME_NAME = "craftly-training-volume"


def locate_sft_checkpoint() -> Path:
    """Auto-locate the latest fine-tuned Craftly SFT model bundle."""
    # 1. Search for SFT checkpoints first
    sft_candidates = [
        Path("/root/craftly_300m_sft_latest.zip"),
        Path("/root/craftly_300m_sft_completed.zip"),
        Path("/mnt/craftly-training-volume/craftly_300m_sft_latest.zip"),
        Path("/mnt/craftly-training-volume/craftly_300m_sft_completed.zip"),
        Path("/vol/craftly_300m_sft_latest.zip"),
        Path("/vol/craftly_300m_sft_completed.zip"),
    ]
    for c in sft_candidates:
        if c.exists():
            return c

    for root in [Path("/root"), Path("/mnt"), Path("/vol")]:
        if root.exists():
            zips = sorted(root.rglob("*craftly*300m*sft*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
            if zips:
                return zips[0]

    # 2. Fall back to extracted model or pretraining foundation checkpoint
    fallback_candidates = [
        Path("/root/model_300m_extracted/model.pt"),
        Path("/mnt/craftly-training-volume/craftly_300m_checkpoint_latest.zip"),
        Path("/mnt/craftly-training-volume/craftly_300m_checkpoint_completed.zip"),
        Path("/vol/craftly_300m_checkpoint_latest.zip"),
    ]
    for c in fallback_candidates:
        if c.exists():
            return c

    for root in [Path("/root"), Path("/mnt"), Path("/vol")]:
        if root.exists():
            base_zips = sorted(root.rglob("*craftly*300m*checkpoint*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
            if base_zips:
                return base_zips[0]

    raise FileNotFoundError("Could not auto-locate Craftly SFT checkpoint. Provide --checkpoint <path>.")


async def execute_agentic_pipeline(
    checkpoint_path: str | None = None,
    tokenizer_path: str | None = None,
    output_report_path: str | None = None,
) -> dict[str, Any]:
    """Run full agentic multi-step verification suite."""
    device = select_torch_device("auto")
    print("=" * 70)
    print("CRAFTLY PHASE 3: AGENTIC TOOL-USE & SANDBOX VERIFICATION")
    print(f"Device: {device.type.upper()}")
    print("=" * 70)

    resolved_ckpt = Path(checkpoint_path) if checkpoint_path else locate_sft_checkpoint()
    print(f"[Setup] Loading Model Checkpoint: {resolved_ckpt}")

    backend = NativeCraftlyModelBackend(
        checkpoint_path=resolved_ckpt,
        tokenizer_path=tokenizer_path,
        device=str(device),
    )
    ident = await backend.identity()
    print(f"[Setup] Loaded {ident['architecture']} ({ident['layers']} layers, vocab: {ident['vocab_size']}) on {ident['device']}")

    runtime = AgenticRuntime(backend=backend)
    scenarios = build_canonical_agentic_scenarios()
    print(f"\n[Scenarios] Executing {len(scenarios)} canonical defensive security tasks...\n")

    results: list[dict[str, Any]] = []

    for i, sc in enumerate(scenarios, 1):
        print("=" * 70)
        print(f"[{i}/{len(scenarios)}] TASK: {sc.scenario_id} ({sc.cwe_id})")
        print(f"Description: {sc.task_description}")
        print("=" * 70)

        with tempfile.TemporaryDirectory(prefix=f"task_{sc.scenario_id}_") as td:
            task_dir = Path(td)
            # Populate initial workspace
            for rel_path, content in sc.initial_workspace.items():
                target = task_dir / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            task = AgenticTask(
                task_id=sc.scenario_id,
                workspace_dir=str(task_dir),
                instruction=sc.task_description,
                verification_command=sc.verification_command,
                max_turns=4,
            )

            report = await runtime.run_task(task)
            print(f"\nOutcome:         {report.status.upper()}")
            print(f"Turns Taken:     {report.turns_taken}")
            print(f"Execution Time:  {report.duration_ms / 1000:.2f}s")
            for st in report.steps:
                print(f"  ▶ Turn {st.turn}:")
                if st.thought:
                    print(f"    Thought:      {st.thought.strip()[:180]}")
                if st.tool_call:
                    print(f"    Tool Call:    {st.tool_call.get('tool')} -> {st.tool_call.get('command')}")
                if st.patch:
                    print(f"    Patch:\n{st.patch.strip()[:240]}")
                if st.explanation:
                    print(f"    Explanation:  {st.explanation.strip()[:180]}")
            if report.verification_stdout:
                print(f"\nVerification Output:\n{report.verification_stdout.strip()}\n")

            results.append(report.model_dump())

    out_file = Path(output_report_path or "/root/craftly_agentic_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print("=" * 70)
    print(f"AGENTIC VERIFICATION COMPLETE! Full report saved to: {out_file}")
    print("=" * 70)
    return {"results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Craftly Phase 3: Agentic Execution & Sandbox Verification")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to Craftly checkpoint zip or pt")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer.json")
    parser.add_argument("--output", type=str, default="/root/craftly_agentic_report.json", help="Path for JSON report")
    args = parser.parse_args()

    asyncio.run(execute_agentic_pipeline(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        output_report_path=args.output,
    ))


if __name__ == "__main__":
    main()
