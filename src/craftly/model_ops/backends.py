"""Serving adapters for Craftly-owned model checkpoints.

Craftly is scratch-first. The only production serving backend here targets a
Craftly checkpoint served locally/privately. The mock backend is a test double
for plumbing checks and is not a model strategy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from src.craftly.model_ops.foundation import CheckpointManifest, sha256_file
from src.craftly.shared.config import load_active_profile
from src.craftly.shared.config_contracts import ActiveModelConfigContract


class ModelBackend:
    name: str = "base"

    async def generate(
        self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 1024
    ) -> str:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None

    async def identity(self) -> dict[str, Any]:
        return {}


class MockModelBackend(ModelBackend):
    name = "mock"

    async def generate(
        self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 1024
    ) -> str:
        return (
            "Mock Craftly response. I inspected the request and would create a minimal, tested patch. "
            f"Prompt: {prompt[:500]}"
        )


class CraftlyServingBackend(ModelBackend):
    name = "craftly_serving"

    def __init__(self, *, endpoint: str, model_name: str, api_key: str | None = None) -> None:
        self.endpoint = _validate_serving_endpoint(endpoint)
        self.model_name = model_name
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=60)

    async def generate(
        self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 1024
    ) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = await self.client.post(
            f"{self.endpoint}/chat/completions",
            headers=headers,
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])

    async def identity(self) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = await self.client.get(
            f"{self.endpoint[:-3]}/health/ready",
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
        if (
            payload.get("status") != "ready"
            or payload.get("scratch_only") is not True
            or payload.get("model_name") != self.model_name
        ):
            raise RuntimeError("Craftly serving readiness identity is invalid")
        for key in ("checkpoint_manifest_sha256", "tokenizer_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get(key) or "")):
                raise RuntimeError(f"Craftly serving readiness is missing {key}")
        return payload

    async def aclose(self) -> None:
        await self.client.aclose()


CraftlyServingBackend = CraftlyServingBackend


def _profile_payload() -> dict[str, Any]:
    payload = load_active_profile()
    profile = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}
    env = payload.get("env") if isinstance(payload.get("env"), dict) else {}
    return {**env, **profile}


def build_active_backend() -> ModelBackend:
    profile = _profile_payload()
    backend = str(
        profile.get("backend")
        or os.environ.get("CRAFTLY_MODEL_BACKEND")
        or os.environ.get("CRAFTLY_MODEL_BACKEND")
        or "mock"
    )
    if backend in {"craftly_serving", "craftly_serving", "active"}:
        return CraftlyServingBackend(
            endpoint=str(
                profile.get("endpoint")
                or os.environ.get("CRAFTLY_MODEL_ENDPOINT")
                or os.environ.get("CRAFTLY_MODEL_ENDPOINT")
                or "http://127.0.0.1:8000/v1"
            ),
            model_name=str(
                profile.get("model_name")
                or os.environ.get("CRAFTLY_MODEL_NAME")
                or os.environ.get("CRAFTLY_MODEL_NAME")
                or "craftly-scratch"
            ),
            api_key=os.environ.get("CRAFTLY_MODEL_API_KEY")
            or os.environ.get("CRAFTLY_MODEL_API_KEY"),
        )
    return MockModelBackend()


def list_model_profiles() -> dict[str, Any]:
    endpoint = os.environ.get("CRAFTLY_MODEL_ENDPOINT") or os.environ.get(
        "CRAFTLY_MODEL_ENDPOINT", "http://127.0.0.1:8000/v1"
    )
    model_name = os.environ.get("CRAFTLY_MODEL_NAME") or os.environ.get(
        "CRAFTLY_MODEL_NAME", "craftly-scratch"
    )
    return {
        "mock": {"backend": "mock", "quality": "test double only, not a real model"},
        "craftly-scratch-local": {
            "backend": "craftly_serving",
            "endpoint": endpoint,
            "model_name": model_name,
            "checkpoint_policy": "Craftly-owned scratch checkpoint only",
        },
    }


def activate_model_profile(name: str, *, run_id: str = "craftly-profile") -> dict[str, Any]:
    profiles = list_model_profiles()
    key = (
        "craftly-scratch-local"
        if name in {"craftly-scratch-local", "craftly-scratch-local"}
        else name
    )
    if key not in profiles:
        raise ValueError(f"unknown model profile: {name}")
    return {"run_id": run_id, "activated": name, "profile": profiles[key]}


def active_model_health() -> dict[str, Any]:
    profile = _profile_payload()
    backend = str(
        profile.get("backend")
        or os.environ.get("CRAFTLY_MODEL_BACKEND")
        or os.environ.get("CRAFTLY_MODEL_BACKEND")
        or "mock"
    )
    return {
        "ok": True,
        "backend": backend,
        "endpoint": str(
            profile.get("endpoint")
            or os.environ.get("CRAFTLY_MODEL_ENDPOINT")
            or os.environ.get("CRAFTLY_MODEL_ENDPOINT")
            or ""
        ),
        "model_name": str(
            profile.get("model_name")
            or os.environ.get("CRAFTLY_MODEL_NAME")
            or os.environ.get("CRAFTLY_MODEL_NAME")
            or "mock"
        ),
    }


def _load_json_object(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.stat().st_size > 100_000_000:
        raise ValueError(f"{label} must be a regular file no larger than 100 MB")
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} contains invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return source, payload


def _verify_checkpoint_manifest(path: str | Path) -> tuple[Path, CheckpointManifest]:
    manifest_path, payload = _load_json_object(path, "checkpoint manifest")
    manifest = CheckpointManifest.model_validate(payload)
    root = Path(manifest.checkpoint_dir).expanduser().resolve(strict=True)
    if not manifest.files:
        raise ValueError("checkpoint manifest contains no files")
    found_model = False
    for entry in manifest.files:
        relative = Path(str(entry.get("path") or ""))
        candidate = (root / relative).resolve(strict=True)
        if relative.is_absolute() or (candidate != root and root not in candidate.parents):
            raise ValueError("checkpoint manifest contains a path outside checkpoint_dir")
        if candidate.stat().st_size != int(entry.get("size_bytes", -1)):
            raise ValueError(f"checkpoint file size changed: {relative.as_posix()}")
        if sha256_file(candidate) != str(entry.get("sha256") or ""):
            raise ValueError(f"checkpoint file hash changed: {relative.as_posix()}")
        found_model = found_model or relative.as_posix() == "model.pt"
    if not found_model:
        raise ValueError("native checkpoint manifest does not contain model.pt")
    return manifest_path, manifest


def _validate_serving_endpoint(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("serving endpoint must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("serving endpoint must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("serving endpoint must not contain a query string or fragment")
    if parsed.path.rstrip("/") != "/v1":
        raise ValueError("serving endpoint path must be exactly /v1")
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("remote serving endpoint must use HTTPS")
    return normalized


def promote_scratch_checkpoint(
    *,
    checkpoint_manifest: str | Path,
    tokenizer_path: str | Path,
    evaluation_report: str | Path,
    output_path: str | Path,
    endpoint: str,
    model_name: str = "craftly-scratch",
    promotion_mode: str = "validation",
    scorecard_report: str | Path | None = None,
) -> ActiveModelConfigContract:
    if promotion_mode not in {"validation", "production"}:
        raise ValueError("promotion_mode must be validation or production")
    manifest_path, manifest = _verify_checkpoint_manifest(checkpoint_manifest)
    tokenizer = Path(tokenizer_path).expanduser().resolve(strict=True)
    if not tokenizer.is_file():
        raise ValueError("tokenizer_path must be a regular file")
    tokenizer_sha256 = sha256_file(tokenizer)
    eval_path, evaluation = _load_json_object(evaluation_report, "evaluation report")
    from src.craftly.evaluation.benchmark_suites import BenchmarkSuitesReport

    evaluation_contract = BenchmarkSuitesReport.model_validate(evaluation)
    if (
        evaluation_contract.status != "passed"
        or evaluation_contract.evaluation_mode != "executable_model"
    ):
        raise ValueError("checkpoint promotion requires a passed executable_model benchmark report")
    if not evaluation_contract.suites:
        raise ValueError("evaluation report contains no executable suites")
    for suite in evaluation_contract.suites:
        if suite.status != "passed" or suite.total < 1 or suite.pass_at_k.get("pass@1", 0.0) <= 0.0:
            raise ValueError(
                f"evaluation suite {suite.name!r} has no positive passed pass@1 result"
            )
        report = suite.report or {}
        bound_manifest = report.get("checkpoint_manifest")
        bound_tokenizer = report.get("tokenizer_path")
        tasks = report.get("tasks")
        if not bound_manifest or Path(str(bound_manifest)).expanduser().resolve() != manifest_path:
            raise ValueError(
                f"evaluation suite {suite.name!r} is not bound to the selected checkpoint"
            )
        if not bound_tokenizer or Path(str(bound_tokenizer)).expanduser().resolve() != tokenizer:
            raise ValueError(
                f"evaluation suite {suite.name!r} is not bound to the selected tokenizer"
            )
        if not isinstance(tasks, list) or len(tasks) != suite.total:
            raise ValueError(f"evaluation suite {suite.name!r} has incomplete task evidence")
        for task in tasks:
            if not isinstance(task, dict):
                raise ValueError(
                    f"evaluation suite {suite.name!r} contains malformed task evidence"
                )
            candidates = task.get("candidates")
            if (
                not isinstance(candidates, list)
                or len(candidates) != int(task.get("candidate_count", 0))
                or not candidates
            ):
                raise ValueError(
                    f"evaluation suite {suite.name!r} contains incomplete candidate evidence"
                )
            if any(
                not isinstance(candidate, dict)
                or not re.fullmatch(r"[0-9a-f]{64}", str(candidate.get("output_sha256") or ""))
                for candidate in candidates
            ):
                raise ValueError(
                    f"evaluation suite {suite.name!r} contains invalid candidate hashes"
                )

    evidence = {
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "tokenizer_sha256": tokenizer_sha256,
        "evaluation_report_sha256": sha256_file(eval_path),
    }
    blockers: list[str] = []
    if promotion_mode == "validation":
        blockers.append(
            "validation checkpoint has not passed the governed 50-task production scorecard"
        )
    else:
        if scorecard_report is None:
            raise ValueError("production promotion requires --scorecard-report")
        scorecard_path, scorecard = _load_json_object(scorecard_report, "scorecard report")
        from src.craftly.evaluation.agent_scorecard import AgentScorecardReport

        scorecard_contract = AgentScorecardReport.model_validate(scorecard)
        if (
            scorecard_contract.status != "passed"
            or scorecard_contract.policy_mode != "strict"
            or not 50 <= scorecard_contract.task_count <= 100
            or len(scorecard_contract.tasks) != scorecard_contract.task_count
        ):
            raise ValueError(
                "production promotion requires a passed scorecard with at least 50 tasks"
            )
        expected_scorecard_evidence = {
            "checkpoint_manifest_sha256": evidence["checkpoint_manifest_sha256"],
            "tokenizer_sha256": evidence["tokenizer_sha256"],
            "evaluation_report_sha256": evidence["evaluation_report_sha256"],
        }
        if any(
            scorecard_contract.model_evidence.get(key) != expected
            for key, expected in expected_scorecard_evidence.items()
        ):
            raise ValueError(
                "production scorecard is not bound to the selected checkpoint, tokenizer, "
                "and executable evaluation report"
            )
        if not re.fullmatch(
            r"[0-9a-f]{64}",
            scorecard_contract.model_evidence.get("serving_identity_sha256", ""),
        ):
            raise ValueError(
                "production scorecard is missing verified live serving identity evidence"
            )
        evidence["scorecard_report_sha256"] = sha256_file(scorecard_path)

    endpoint_value = _validate_serving_endpoint(endpoint)
    profile = {
        "name": f"{model_name}-{promotion_mode}",
        "kind": "local"
        if urlparse(endpoint_value).hostname in {"127.0.0.1", "localhost", "::1"}
        else "remote",
        "family": "craftly-scratch",
        "size_class": manifest.architecture_name,
        "backend": "craftly_serving",
        "model_name": model_name,
        "endpoint": endpoint_value,
        "checkpoint_manifest": str(manifest_path),
        "tokenizer_path": str(tokenizer),
        "requires_cuda": False,
        "dev_only": False,
        "scratch_only": True,
        "evidence": evidence,
        "notes": [
            f"Promotion mode: {promotion_mode}.",
            f"Checkpoint step: {manifest.step}; trained tokens: {manifest.trained_tokens}.",
        ],
    }
    contract = ActiveModelConfigContract.model_validate(
        {
            "schema_version": 2,
            "profile": profile,
            "env": {
                "CRAFTLY_ACTIVE_PROFILE": profile["name"],
                "CRAFTLY_MODEL_BACKEND": "craftly_serving",
                "CRAFTLY_MODEL_ENDPOINT": endpoint_value,
                "CRAFTLY_MODEL_NAME": model_name,
                "CRAFTLY_CHECKPOINT_MANIFEST": str(manifest_path),
                "CRAFTLY_TOKENIZER_PATH": str(tokenizer),
            },
            "run_id": f"craftly-promotion-{uuid.uuid4()}",
            "production_blockers": blockers,
        }
    )
    target = Path(output_path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite active model promotion artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(contract.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        delete=False,
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return contract


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage Craftly scratch-model backend promotion.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote = subparsers.add_parser(
        "promote-checkpoint",
        help="Create an immutable active-profile artifact from measured checkpoint evidence.",
    )
    promote.add_argument("--checkpoint-manifest", required=True)
    promote.add_argument("--tokenizer-path", required=True)
    promote.add_argument("--evaluation-report", required=True)
    promote.add_argument("--scorecard-report")
    promote.add_argument("--output", required=True)
    promote.add_argument("--endpoint", required=True)
    promote.add_argument("--model-name", default="craftly-scratch")
    promote.add_argument(
        "--promotion-mode", choices=["validation", "production"], default="validation"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command != "promote-checkpoint":
        raise ValueError(f"unsupported command: {args.command}")
    report = promote_scratch_checkpoint(
        checkpoint_manifest=args.checkpoint_manifest,
        tokenizer_path=args.tokenizer_path,
        evaluation_report=args.evaluation_report,
        scorecard_report=args.scorecard_report,
        output_path=args.output,
        endpoint=args.endpoint,
        model_name=args.model_name,
        promotion_mode=args.promotion_mode,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
