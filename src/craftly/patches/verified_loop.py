"""Repository-aware patch verification loop.

This is the production-facing glue between repository indexing, context
packing, patch preview/apply, verifier execution, and post-patch reindexing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.db import LocalStore
from src.craftly.indexing import ContextBuilder, RepositoryIndexer
from src.craftly.patches.service import FileEdit, PatchService, PatchVerifyRequest
from src.craftly.shared.schemas import StrictModel


class RepositoryPatchLoopRequest(StrictModel):
    repo_path: str
    goal: str
    edits: list[FileEdit] = Field(min_length=1)
    commands: list[list[str]] = Field(default_factory=list)
    project_name: str = "patch-loop"
    token_budget: int = Field(default=24_000, ge=512)
    run_secret_scan: bool = True
    run_semgrep: bool = False
    run_codeql: bool = False
    fail_on_tool_unavailable: bool = False
    apply_on_accept: bool = False
    store_path: str | None = None


class RepositoryPatchLoopReport(StrictModel):
    status: str
    verdict: str
    project_id: str
    repo_path: str
    goal: str
    initial_index: dict[str, Any]
    pre_patch_context: dict[str, Any]
    patch_verification: dict[str, Any]
    post_patch_index: dict[str, Any]
    post_patch_context: dict[str, Any]
    files_changed: list[str]
    duration_ms: float
    created_at_unix: float = Field(default_factory=time.time)


def run_repository_patch_loop(request: RepositoryPatchLoopRequest) -> RepositoryPatchLoopReport:
    started = time.perf_counter()
    repo = Path(request.repo_path).resolve()
    if not repo.exists() or not repo.is_dir():
        raise FileNotFoundError(f"repo_path is not a directory: {repo}")
    with LocalStore(request.store_path) as store:
        project = store.create_project(name=request.project_name, repo_path=str(repo))
        indexer = RepositoryIndexer(store)
        context = ContextBuilder(store)
        initial_index = indexer.index_project(project_id=project["id"])
        changed_files = [edit.path for edit in request.edits]
        pre_context = context.build(
            project_id=project["id"],
            query=request.goal,
            pinned_files=changed_files,
            token_budget=request.token_budget,
        )
        patch_response = PatchService(store).preview_apply_verify(
            PatchVerifyRequest(
                project_id=project["id"],
                edits=request.edits,
                commands=request.commands,
                run_secret_scan=request.run_secret_scan,
                run_semgrep=request.run_semgrep,
                run_codeql=request.run_codeql,
                fail_on_tool_unavailable=request.fail_on_tool_unavailable,
                apply_on_accept=request.apply_on_accept,
            )
        )
        post_index = indexer.index_project(project_id=project["id"])
        post_context = context.build(
            project_id=project["id"],
            query=request.goal,
            pinned_files=changed_files,
            token_budget=request.token_budget,
        )
    verdict = patch_response.verdict
    return RepositoryPatchLoopReport(
        status="passed" if verdict == "accept" else "failed",
        verdict=verdict,
        project_id=project["id"],
        repo_path=str(repo),
        goal=request.goal,
        initial_index=initial_index.model_dump(),
        pre_patch_context=pre_context.model_dump(),
        patch_verification=patch_response.model_dump(),
        post_patch_index=post_index.model_dump(),
        post_patch_context=post_context.model_dump(),
        files_changed=changed_files,
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
    )


def _load_edits(path: str | Path) -> list[FileEdit]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("edits") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("edits file must be a JSON list or {'edits': [...]}")
    return [FileEdit.model_validate(row) for row in rows]


def _parse_command(raw: str) -> list[str]:
    import shlex

    command = shlex.split(raw, posix=False)
    if not command:
        raise ValueError("empty command")
    return command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repository-aware patch verification loop.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--edits-json", required=True)
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--store-path")
    parser.add_argument("--apply-on-accept", action="store_true")
    parser.add_argument("--run-semgrep", action="store_true")
    parser.add_argument("--run-codeql", action="store_true")
    parser.add_argument("--fail-on-tool-unavailable", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_repository_patch_loop(
        RepositoryPatchLoopRequest(
            repo_path=args.repo,
            goal=args.goal,
            edits=_load_edits(args.edits_json),
            commands=[_parse_command(item) for item in args.command],
            store_path=args.store_path,
            apply_on_accept=args.apply_on_accept,
            run_semgrep=args.run_semgrep,
            run_codeql=args.run_codeql,
            fail_on_tool_unavailable=args.fail_on_tool_unavailable,
        )
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
    if report.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
