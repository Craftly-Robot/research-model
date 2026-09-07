"""Direct-kernel Kaggle/Colab validation launcher for Craftly.

Recommended notebook invocation:

    %run -i deploy/gpu/run_workspace_validation.py --profile defensive-1k

The launcher never accepts a shell command. It resolves an immutable workspace
profile, constructs the bounded pipeline arguments, and invokes the Python
pipeline in the active notebook kernel so progress is visible immediately.
When workspace credentials are available it also creates and claims a tracked
job and sends live events through the standard WAL-backed progress reporter.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.gpu.run_real_data_training_pipeline import (  # noqa: E402
    parse_args as parse_pipeline_args,
    run as run_pipeline,
)
from src.craftly.training_client import Workspace  # noqa: E402
from src.craftly.training_workspace import TrainingProfile, TrainingProfileRegistry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an immutable Craftly notebook validation profile in the active kernel.")
    parser.add_argument("--profile", default="defensive-1k")
    parser.add_argument("--work-dir")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--sources", default="config/data_sources.ultimate.json")
    parser.add_argument("--standalone", action="store_true", help="Do not register this validation with a remote workspace.")
    return parser.parse_args()


def _bounded_override(profile: TrainingProfile, name: str, value: int | None) -> int | None:
    if value is None:
        return None
    bounds = profile.allowed_overrides.get(name)
    if bounds is None:
        raise ValueError(f"profile {profile.profile_id} does not permit --{name.replace('_', '-')}")
    if not bounds.minimum <= value <= bounds.maximum:
        raise ValueError(f"{name} must be between {bounds.minimum} and {bounds.maximum}")
    return value


def _pipeline_namespace(profile: TrainingProfile, args: argparse.Namespace) -> argparse.Namespace:
    if profile.scheduler != "notebook" or not profile.dev_only or profile.run_type != "data_pipeline":
        raise ValueError("direct notebook execution is restricted to dev-only notebook data-pipeline profiles")
    steps = _bounded_override(profile, "steps", args.steps) or profile.steps
    max_docs = _bounded_override(profile, "max_docs", args.max_docs)
    output_dir = args.work_dir or f"artifacts/craftly/workspace-validation/{profile.profile_id}-{int(time.time())}"
    argv = [
        "run_real_data_training_pipeline.py",
        "--sources",
        args.sources,
        "--work-dir",
        output_dir,
        "--kaggle-validation",
        "--model-profile",
        profile.model_profile,
        "--curriculum-mode",
        profile.curriculum_mode,
        "--steps",
        str(steps),
        "--sequence-length",
        str(profile.sequence_length),
        "--batch-size",
        str(profile.batch_size),
        "--gradient-accumulation-steps",
        str(profile.gradient_accumulation_steps),
        "--dtype",
        profile.dtype,
        "--progress-to-stdout",
        "--progress-every-docs",
        "10",
        "--progress-every-steps",
        "1",
    ]
    if max_docs is not None:
        argv.extend(["--max-docs", str(max_docs)])
    original = sys.argv
    try:
        sys.argv = argv
        return parse_pipeline_args()
    finally:
        sys.argv = original


async def _register_workspace_job(
    workspace: Workspace,
    profile: TrainingProfile,
    args: argparse.Namespace,
) -> tuple[str, str]:
    overrides = {key: value for key, value in {"steps": args.steps, "max_docs": args.max_docs}.items() if value is not None}
    run = await workspace.train(
        profile.profile_id,
        follow=False,
        idempotency_key=f"notebook-{profile.profile_id}-{uuid.uuid4()}",
        overrides=overrides,
        metadata={"client": "notebook-direct-kernel", "validation_only": True},
    )
    claim = await workspace.claim_notebook_job(run.job_id)
    attempt_id = str(claim["attempt"]["attempt_id"])
    os.environ.update(
        {
            "CRAFTLY_TRAINING_JOB_ID": run.job_id,
            "CRAFTLY_TRAINING_ATTEMPT_ID": attempt_id,
            "CRAFTLY_WORKSPACE_ACCESS_TOKEN": str(claim["worker_access_token"]),
        }
    )
    print(f"[craftly-workspace] claimed job={run.job_id} attempt={attempt_id}", flush=True)
    return run.job_id, attempt_id


async def run(args: argparse.Namespace) -> dict[str, Any]:
    profile = TrainingProfileRegistry.from_file().latest(args.profile)
    pipeline_args = _pipeline_namespace(profile, args)
    workspace: Workspace | None = None
    job_id = ""
    attempt_id = ""
    workspace_configured = bool(os.environ.get("CRAFTLY_WORKSPACE_URL") and os.environ.get("CRAFTLY_BOOTSTRAP_TOKEN"))
    if workspace_configured and not args.standalone:
        workspace = Workspace.from_environment()
        job_id, attempt_id = await _register_workspace_job(workspace, profile, args)
    elif not args.standalone:
        print(
            "[craftly-workspace] remote tracking disabled: set CRAFTLY_WORKSPACE_URL and CRAFTLY_BOOTSTRAP_TOKEN; running standalone validation",
            flush=True,
        )
    print(
        json.dumps(
            {
                "event": "craftly_notebook_validation_start",
                "profile": profile.profile_id,
                "profile_hash": profile.immutable_hash,
                "work_dir": pipeline_args.output_dir,
                "steps": pipeline_args.train_steps,
                "tracked_job_id": job_id or None,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        result = await asyncio.wait_for(
            run_pipeline(pipeline_args),
            timeout=profile.runtime_limits.maximum_wall_time_seconds,
        )
        if workspace and result.get("status") != "complete":
            await workspace.emit_events(
                job_id,
                attempt_id,
                [
                    {
                        "source_sequence": 2_000_000_000,
                        "kind": "error",
                        "stage": "pipeline",
                        "status": "blocked",
                        "message": str(result.get("block_reason") or "validation pipeline blocked"),
                        "payload": {"failure_class": "quality_gate"},
                    }
                ],
            )
        return result
    except Exception as exc:
        if workspace and job_id and attempt_id:
            await workspace.emit_events(
                job_id,
                attempt_id,
                [
                    {
                        "source_sequence": 2_000_000_001,
                        "kind": "error",
                        "stage": "pipeline",
                        "status": "failed",
                        "message": str(exc),
                        "payload": {"failure_class": "runtime"},
                    }
                ],
            )
        raise
    finally:
        if workspace:
            await workspace.close()


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run(args))
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    raise SystemExit(0 if payload.get("status") == "complete" else 2)


if __name__ == "__main__":
    main()
