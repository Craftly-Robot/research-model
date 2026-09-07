"""Native MVP patch preview/apply/rollback service."""

from __future__ import annotations

import difflib
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from src.craftly.db import LocalStore
from src.craftly.indexing import RepositoryIndexer
from src.craftly.shared.schemas import StrictModel
from src.craftly.tools.policy import project_root
from src.craftly.verifier import VerificationRequest, VerifierRuntime


class FileEdit(StrictModel):
    path: str = Field(min_length=1, max_length=1024)
    new_content: str

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        parts = normalized.split("/")
        if (
            normalized.startswith("/")
            or Path(normalized).drive
            or (len(parts[0]) >= 2 and parts[0][1] == ":")
            or any(part in {"", ".", ".."} for part in parts)
            or any(ord(character) < 32 for character in normalized)
        ):
            raise ValueError(f"unsafe edit path: {value}")
        if normalized == ".git" or normalized.startswith(".git/"):
            raise ValueError("patches cannot write inside .git")
        return normalized


class PatchPreviewRequest(StrictModel):
    project_id: str
    run_id: str | None = None
    edits: list[FileEdit] = Field(min_length=1)


class PatchResponse(StrictModel):
    patch_id: str
    project_id: str
    run_id: str | None = None
    status: str
    diff: str
    files_changed: list[str]


class PatchVerifyRequest(PatchPreviewRequest):
    commands: list[list[str]] = Field(default_factory=list)
    run_secret_scan: bool = True
    run_semgrep: bool = False
    run_codeql: bool = False
    fail_on_tool_unavailable: bool = False
    apply_on_accept: bool = False


class PatchVerifyResponse(StrictModel):
    patch: PatchResponse
    verification: dict[str, Any]
    verdict: str
    final_status: str
    rolled_back: bool


def resolve_inside(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes project root: {relative}")
    return target


class PatchService:
    def __init__(self, store: LocalStore | None = None) -> None:
        self.store = store or LocalStore()

    def preview(self, request: PatchPreviewRequest) -> PatchResponse:
        root = project_root(self.store, request.project_id)
        diff_parts: list[str] = []
        backup: dict[str, Any] = {"files": {}}
        for edit in request.edits:
            target = resolve_inside(root, edit.path)
            before = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
            backup["files"][edit.path] = {
                "existed": target.exists(),
                "content": before,
            }
            diff_parts.append(self.make_diff(edit.path, before, edit.new_content))
        record = self.store.create_patch_record(
            project_id=request.project_id,
            run_id=request.run_id,
            status="preview",
            diff="\n".join(part for part in diff_parts if part),
            files_changed=[edit.path for edit in request.edits],
            backup={**backup, "edits": [edit.model_dump() for edit in request.edits]},
        )
        return self.response_from_record(record)

    def apply(self, patch_id: str) -> PatchResponse:
        record = self.store.get_patch(patch_id)
        if record is None:
            raise KeyError(f"unknown patch: {patch_id}")
        if record["status"] == "applied":
            return self.response_from_record(record)
        if record["status"] != "preview":
            raise ValueError(f"patch {patch_id} cannot be applied from status {record['status']}")
        root = project_root(self.store, record["project_id"])
        written: list[Path] = []
        try:
            for edit in record["backup"].get("edits", []):
                target = resolve_inside(root, edit["path"])
                self._atomic_write(target, str(edit["new_content"]))
                written.append(target)
        except Exception:
            self._restore_files(root, record["backup"].get("files", {}), only=written)
            raise
        self.store.update_patch_status(patch_id, "applied", applied=True)
        return self.response_from_record(self.store.get_patch(patch_id) or record)

    def rollback(self, patch_id: str) -> PatchResponse:
        record = self.store.get_patch(patch_id)
        if record is None:
            raise KeyError(f"unknown patch: {patch_id}")
        if record["status"] == "rolled_back":
            return self.response_from_record(record)
        if record["status"] not in {"preview", "applied"}:
            raise ValueError(f"patch {patch_id} cannot be rolled back from status {record['status']}")
        # A preview has never mutated the workspace. Rejecting one must not
        # replace files or change their timestamps and permissions.
        if record["status"] == "applied":
            root = project_root(self.store, record["project_id"])
            self._restore_files(root, record["backup"].get("files", {}))
        self.store.update_patch_status(patch_id, "rolled_back", rolled_back=True)
        return self.response_from_record(self.store.get_patch(patch_id) or record)

    def _restore_files(
        self,
        root: Path,
        backups: dict[str, Any],
        *,
        only: list[Path] | None = None,
    ) -> None:
        selected = {path.resolve() for path in only} if only is not None else None
        for relative, backup in backups.items():
            target = resolve_inside(root, relative)
            if selected is not None and target.resolve() not in selected:
                continue
            # Older patch records stored the original content directly.
            if isinstance(backup, dict):
                existed = bool(backup.get("existed"))
                content = str(backup.get("content") or "")
            else:
                existed = True
                content = str(backup)
            if not existed:
                target.unlink(missing_ok=True)
                self._remove_empty_parents(target.parent, root)
                continue
            self._atomic_write(target, content)

    @staticmethod
    def _atomic_write(target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o600
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".craftly-tmp",
            dir=str(target.parent),
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, target)
            if os.name != "nt":
                directory_descriptor = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _remove_empty_parents(path: Path, root: Path) -> None:
        while path != root:
            try:
                path.rmdir()
            except OSError:
                return
            path = path.parent

    def preview_apply_verify(self, request: PatchVerifyRequest) -> PatchVerifyResponse:
        patch = self.preview(request)
        applied = self.apply(patch.patch_id)
        RepositoryIndexer(self.store).index_project(project_id=request.project_id)
        verification = VerifierRuntime(self.store).run(
            VerificationRequest(
                project_id=request.project_id,
                run_id=request.run_id,
                patch_id=patch.patch_id,
                commands=request.commands,
                run_secret_scan=request.run_secret_scan,
                run_semgrep=request.run_semgrep,
                run_codeql=request.run_codeql,
                fail_on_tool_unavailable=request.fail_on_tool_unavailable,
            )
        )
        accepted = verification.verdict == "accept"
        rolled_back = False
        final_patch = applied
        if not accepted or not request.apply_on_accept:
            final_patch = self.rollback(patch.patch_id)
            rolled_back = True
            RepositoryIndexer(self.store).index_project(project_id=request.project_id)
        return PatchVerifyResponse(
            patch=final_patch,
            verification=verification.model_dump(),
            verdict=verification.verdict,
            final_status=final_patch.status,
            rolled_back=rolled_back,
        )

    def response_from_record(self, record: dict[str, Any]) -> PatchResponse:
        return PatchResponse(
            patch_id=record["id"],
            project_id=record["project_id"],
            run_id=record.get("run_id"),
            status=record["status"],
            diff=record["diff"],
            files_changed=record["files_changed"],
        )

    def make_diff(self, path: str, before: str, after: str) -> str:
        return "\n".join(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"{path}:before",
                tofile=f"{path}:after",
                lineterm="",
            )
        )

