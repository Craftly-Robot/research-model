"""MVP repository indexing engine.

This module gives Craftly a Cursor-style local repository intelligence baseline:
file inventory, content hashes, language detection, chunking, symbol extraction,
and storage into the MVP database. It intentionally works without Qdrant so the
system is testable on day one; vector storage can subscribe to the same chunks.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol

from pydantic import Field

from src.craftly.db.local_store import (
    LocalStore,
    PostgresRAGDispatcher,
    PostgresRAGStore,
    PostgresRAGStoreFactory,
)
from src.craftly.learning.storage import ObjectStore, ObjectStoreConfig, create_object_store
from src.craftly.shared.schemas import StrictModel


LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".sh": "bash",
    ".bash": "bash",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
}

DEFAULT_EXCLUDES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "target",
    ".next",
    ".turbo",
    ".idea",
    ".vscode",
}

SYMBOL_REGEX = re.compile(
    r"^\s*(?:async\s+)?(?:def|class|function|func|fn|struct|enum|interface|type)\s+([A-Za-z_][\w]*)",
    re.MULTILINE,
)

IMPORT_REGEX_BY_LANGUAGE = {
    "javascript": re.compile(r"^\s*import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]|^\s*const\s+.+?=\s+require\(['\"]([^'\"]+)['\"]\)", re.MULTILINE),
    "typescript": re.compile(r"^\s*import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]|^\s*const\s+.+?=\s+require\(['\"]([^'\"]+)['\"]\)", re.MULTILINE),
    "go": re.compile(r"^\s*import\s+(?:\(\s*)?\"([^\"]+)\"", re.MULTILINE),
    "rust": re.compile(r"^\s*use\s+([^;]+);", re.MULTILINE),
    "java": re.compile(r"^\s*import\s+([^;]+);", re.MULTILINE),
    "cpp": re.compile(r"^\s*#\s*include\s+[<\"]([^>\"]+)[>\"]", re.MULTILINE),
    "c": re.compile(r"^\s*#\s*include\s+[<\"]([^>\"]+)[>\"]", re.MULTILINE),
    "bash": re.compile(r"^\s*(?:source|\.)\s+(.+)$", re.MULTILINE),
}

CHUNKER_VERSION = "craftly-code-chunker-v3"

TREE_SITTER_LANGUAGES = {
    "javascript",
    "typescript",
    "go",
    "rust",
    "java",
    "c",
    "cpp",
    "bash",
}

TREE_SITTER_SYMBOL_TYPES: dict[str, dict[str, str]] = {
    "javascript": {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
    },
    "typescript": {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "type",
    },
    "rust": {
        "function_item": "function",
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "trait",
        "impl_item": "implementation",
    },
    "java": {
        "method_declaration": "method",
        "constructor_declaration": "constructor",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
    },
    "c": {
        "function_definition": "function",
        "struct_specifier": "struct",
        "enum_specifier": "enum",
    },
    "cpp": {
        "function_definition": "function",
        "class_specifier": "class",
        "struct_specifier": "struct",
        "enum_specifier": "enum",
        "namespace_definition": "namespace",
    },
    "bash": {"function_definition": "function"},
}

TREE_SITTER_CALL_TYPES = {
    "call_expression",
    "function_call",
    "command",
    "macro_invocation",
}

TREE_SITTER_MUTATION_TYPES = {
    "assignment_expression",
    "assignment_statement",
    "augmented_assignment_expression",
    "update_expression",
    "let_declaration",
    "variable_declaration",
    "short_var_declaration",
}


@dataclass(frozen=True)
class PythonModuleFacts:
    imports: list[str]
    module_calls: list[str]
    module_mutations: list[str]


@lru_cache(maxsize=len(TREE_SITTER_LANGUAGES))
def tree_sitter_parser(language: str) -> Any:
    """Load a supported parser once and fail closed to the line fallback.

    Parser availability is deliberately optional in development. Production
    qualification checks the parser coverage recorded in each index manifest.
    """

    if language not in TREE_SITTER_LANGUAGES:
        raise ValueError(f"unsupported Tree-sitter language: {language}")
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError as exc:  # pragma: no cover - readiness catches this.
        raise RuntimeError("tree-sitter-language-pack is unavailable") from exc
    parser_name = "typescript" if language == "typescript" else language
    return get_parser(parser_name)


def ts_value(value: Any) -> Any:
    return value() if callable(value) else value


def ts_node_type(node: Any) -> str:
    value = getattr(node, "type", None)
    if value is None:
        value = getattr(node, "kind", None)
    return str(ts_value(value) or "")


def ts_children(node: Any, *, named: bool = False) -> list[Any]:
    attribute = getattr(node, "named_children" if named else "children", None)
    if attribute is not None:
        return list(ts_value(attribute) or [])
    count_name = "named_child_count" if named else "child_count"
    child_name = "named_child" if named else "child"
    count = int(ts_value(getattr(node, count_name, 0)) or 0)
    child_getter = getattr(node, child_name)
    return [child_getter(index) for index in range(count)]


def ts_byte(node: Any, boundary: str) -> int:
    return int(ts_value(getattr(node, f"{boundary}_byte")))


def ts_row(node: Any, boundary: str) -> int:
    point = getattr(node, f"{boundary}_point", None)
    if point is None:
        point = getattr(node, f"{boundary}_position")
    point = ts_value(point)
    if hasattr(point, "row"):
        return int(point.row)
    return int(point[0])


def ts_parent(node: Any) -> Any | None:
    return ts_value(getattr(node, "parent", None))


def ts_has_error(node: Any) -> bool:
    return bool(ts_value(getattr(node, "has_error", False)))


class IndexReport(StrictModel):
    project_id: str
    repo_path: str
    status: str
    file_count: int
    chunk_count: int
    symbol_count: int
    revision_id: str
    source_revision: str
    source_snapshot_sha256: str
    chunker_version: str = CHUNKER_VERSION
    changed_file_count: int = 0
    removed_file_count: int = 0
    duration_ms: float
    errors: list[str] = Field(default_factory=list)


class IndexCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_path: str
    language: str
    content: str
    content_hash: str
    size_bytes: int


def stable_uuid(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class RepositoryIndexer:
    def __init__(
        self,
        store: LocalStore | None = None,
        *,
        object_store: ObjectStore | None = None,
        production_mode: bool | None = None,
    ) -> None:
        self.store = store or LocalStore()
        self.production_mode = (
            production_mode
            if production_mode is not None
            else os.environ.get("CRAFTLY_ENV", "development").lower() == "production"
        )
        self.object_store = object_store
        if self.production_mode and self.object_store is None:
            uri = os.environ.get("CRAFTLY_RAG_OBJECT_STORE_URI") or os.environ.get("CRAFTLY_OBJECT_STORE_URI")
            if not uri or not uri.startswith("s3://"):
                raise RuntimeError("production repository indexing requires S3/MinIO snapshot storage")
            self.object_store = create_object_store(
                ObjectStoreConfig(
                    uri=uri,
                    endpoint_url=os.environ.get("CRAFTLY_OBJECT_STORE_ENDPOINT_URL"),
                )
            )

    def index_project(
        self,
        *,
        project_id: str,
        include_suffixes: set[str] | None = None,
        max_file_bytes: int = 1_000_000,
        max_chunk_lines: int = 120,
        max_chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        organization_id: str | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> IndexReport:
        started = time.perf_counter()
        project = (
            self.store.require_project_access(project_id, organization_id)
            if organization_id is not None
            else self.store.get_project(project_id)
        )
        if project is None:
            raise KeyError(f"unknown project: {project_id}")
        repo_path = Path(str(project["repo_path"])).resolve()
        if not repo_path.exists() or not repo_path.is_dir():
            raise FileNotFoundError(f"project repo_path is not a directory: {repo_path}")

        errors: list[str] = []
        if max_chunk_tokens < 128 or max_chunk_tokens > 4096:
            raise ValueError("max_chunk_tokens must be between 128 and 4096")
        if overlap_tokens < 0 or overlap_tokens >= max_chunk_tokens:
            raise ValueError("overlap_tokens must be non-negative and smaller than max_chunk_tokens")

        source_files = list(
            self.iter_source_files(
                repo_path,
                include_suffixes=include_suffixes,
                max_file_bytes=max_file_bytes,
            )
        )
        source_snapshot_sha256 = self.source_snapshot_sha256(source_files)
        source_revision = self.resolve_source_revision(repo_path, source_snapshot_sha256)
        snapshot_object = self.persist_source_snapshot(
            organization_id=str(project["organization_id"]),
            project_id=project_id,
            snapshot_sha256=source_snapshot_sha256,
            source_files=source_files,
        )
        previous_files = self.store.list_workspace_file_hashes(project_id)
        current_hashes = {item.relative_path: item.content_hash for item in source_files}
        changed_file_count = sum(
            1 for path, content_hash in current_hashes.items() if previous_files.get(path) != content_hash
        )
        removed_file_count = len(set(previous_files) - set(current_hashes))
        manifest_base = {
            "schema_version": 1,
            "organization_id": str(project["organization_id"]),
            "project_id": str(project_id),
            "source_revision": source_revision,
            "source_snapshot_sha256": source_snapshot_sha256,
            "chunker_version": CHUNKER_VERSION,
            "max_chunk_tokens": max_chunk_tokens,
            "overlap_tokens": overlap_tokens,
            "source_snapshot_object": snapshot_object,
        }
        revision = self.store.begin_index_revision(
            project_id=project_id,
            source_revision=source_revision,
            source_snapshot_sha256=source_snapshot_sha256,
            chunker_version=CHUNKER_VERSION,
            manifest=manifest_base,
        )
        revision_id = str(revision["id"])
        files: list[dict[str, Any]] = []
        all_chunks: list[dict[str, Any]] = []
        try:
            for source_file in source_files:
                if cancellation_check is not None and cancellation_check():
                    raise IndexCancelled("repository indexing was cancelled")
                file_id = stable_uuid(
                    project["organization_id"], project_id, source_file.relative_path,
                    source_file.content_hash, CHUNKER_VERSION,
                )
                files.append(
                    {
                        "id": file_id,
                        "path": source_file.relative_path,
                        "language": source_file.language,
                        "content_hash": source_file.content_hash,
                        "size_bytes": source_file.size_bytes,
                    }
                )
                chunks = list(
                    self.chunk_file(
                        organization_id=str(project["organization_id"]),
                        project_id=project_id,
                        file_id=file_id,
                        source_file=source_file,
                        max_chunk_lines=max_chunk_lines,
                        max_chunk_tokens=max_chunk_tokens,
                        overlap_tokens=overlap_tokens,
                    )
                )
                all_chunks.extend(chunks)
            graph_report = self.resolve_interprocedural_graph(all_chunks)
            if cancellation_check is not None and cancellation_check():
                raise IndexCancelled("repository indexing was cancelled before revision commit")
            manifest = {
                **manifest_base,
                "file_count": len(files),
                "chunk_count": len(all_chunks),
                "symbol_count": sum(1 for chunk in all_chunks if chunk.get("symbol_name")),
                "parser_counts": dict(
                    sorted(
                        {
                            parser: sum(
                                1
                                for chunk in all_chunks
                                if str((chunk.get("metadata") or {}).get("parser") or "unknown") == parser
                            )
                            for parser in {
                                str((chunk.get("metadata") or {}).get("parser") or "unknown")
                                for chunk in all_chunks
                            }
                        }.items()
                    )
                ),
                "call_graph": graph_report,
                "changed_file_count": changed_file_count,
                "removed_file_count": removed_file_count,
            }
            manifest["manifest_sha256"] = sha256_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
            self.store.commit_index_revision(
                revision_id=revision_id,
                files=files,
                chunks=all_chunks,
                manifest=manifest,
            )
        except Exception as exc:
            try:
                self.store.fail_index_revision(revision_id, str(exc))
            except RuntimeError:
                pass
            raise
        return IndexReport(
            project_id=project_id,
            repo_path=str(repo_path),
            status="ready",
            file_count=len(files),
            chunk_count=len(all_chunks),
            symbol_count=sum(1 for chunk in all_chunks if chunk.get("symbol_name")),
            revision_id=revision_id,
            source_revision=source_revision,
            source_snapshot_sha256=source_snapshot_sha256,
            changed_file_count=changed_file_count,
            removed_file_count=removed_file_count,
            duration_ms=(time.perf_counter() - started) * 1000,
            errors=errors,
        )

    @staticmethod
    def source_snapshot_sha256(source_files: list[SourceFile]) -> str:
        digest = hashlib.sha256()
        for source_file in sorted(source_files, key=lambda item: item.relative_path):
            digest.update(source_file.relative_path.encode("utf-8", "surrogatepass"))
            digest.update(b"\x00")
            digest.update(bytes.fromhex(source_file.content_hash))
        return digest.hexdigest()

    @staticmethod
    def resolve_source_revision(repo_path: Path, snapshot_sha256: str) -> str:
        executable = shutil.which("git")
        if executable is None:
            return f"snapshot:{snapshot_sha256}"
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
        try:
            result = subprocess.run(  # nosec B603 - fixed executable and fixed argument vector.
                [executable, "-C", str(repo_path), "rev-parse", "--verify", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
                env=environment,
            )
            commit = result.stdout.strip().lower()
            if re.fullmatch(r"[0-9a-f]{40,64}", commit):
                return f"git:{commit}"
        except (OSError, subprocess.SubprocessError):
            pass
        return f"snapshot:{snapshot_sha256}"

    def persist_source_snapshot(
        self,
        *,
        organization_id: str,
        project_id: str,
        snapshot_sha256: str,
        source_files: list[SourceFile],
    ) -> dict[str, Any] | None:
        if self.object_store is None:
            return None
        key = f"rag/snapshots/{organization_id}/{project_id}/{snapshot_sha256}.tar"
        with tempfile.TemporaryDirectory(prefix="craftly-rag-snapshot-") as directory:
            archive = Path(directory) / "snapshot.tar"
            with tarfile.open(archive, mode="w", format=tarfile.PAX_FORMAT) as handle:
                manifest_rows: list[dict[str, Any]] = []
                for source_file in sorted(source_files, key=lambda item: item.relative_path):
                    payload = source_file.content.encode("utf-8", "surrogatepass")
                    info = tarfile.TarInfo(name=source_file.relative_path)
                    info.size = len(payload)
                    info.mode = 0o600
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    handle.addfile(info, io.BytesIO(payload))
                    manifest_rows.append(
                        {
                            "path": source_file.relative_path,
                            "content_hash": source_file.content_hash,
                            "size_bytes": source_file.size_bytes,
                        }
                    )
                manifest_payload = json.dumps(
                    {
                        "schema_version": 1,
                        "snapshot_sha256": snapshot_sha256,
                        "files": manifest_rows,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                info = tarfile.TarInfo(name=".craftly-snapshot-manifest.json")
                info.size = len(manifest_payload)
                info.mode = 0o600
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                handle.addfile(info, io.BytesIO(manifest_payload))
            stored = self.object_store.put_file(archive, key=key)
        return {
            "uri": stored.uri,
            "sha256": stored.sha256,
            "size_bytes": stored.size_bytes,
            "snapshot_sha256": snapshot_sha256,
        }

    def iter_source_files(
        self,
        repo_path: Path,
        *,
        include_suffixes: set[str] | None,
        max_file_bytes: int,
    ) -> Iterator[SourceFile]:
        suffixes = include_suffixes or set(LANGUAGE_BY_SUFFIX)
        for path in sorted(repo_path.rglob("*")):
            if not path.is_file():
                continue
            if any(part in DEFAULT_EXCLUDES for part in path.relative_to(repo_path).parts):
                continue
            suffix = path.suffix.lower()
            if suffix not in suffixes or suffix not in LANGUAGE_BY_SUFFIX:
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if len(raw) > max_file_bytes or b"\x00" in raw[:4096]:
                continue
            text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
            yield SourceFile(
                path=path,
                relative_path=path.relative_to(repo_path).as_posix(),
                language=LANGUAGE_BY_SUFFIX[suffix],
                content=text,
                content_hash=hashlib.sha256(raw).hexdigest(),
                size_bytes=len(raw),
            )

    def chunk_file(
        self,
        *,
        organization_id: str,
        project_id: str,
        file_id: str,
        source_file: SourceFile,
        max_chunk_lines: int,
        max_chunk_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> Iterable[dict[str, Any]]:
        if source_file.language == "python":
            chunks = list(self.python_symbol_chunks(source_file))
            if chunks:
                for chunk in chunks:
                    for bounded in self.bound_chunk(chunk, max_chunk_tokens=max_chunk_tokens, overlap_tokens=overlap_tokens):
                        yield self.chunk_payload(organization_id, project_id, file_id, source_file, **bounded)
                return
        if source_file.language in TREE_SITTER_LANGUAGES:
            if self.production_mode:
                # A production index must never silently lose symbol/call-graph
                # fidelity because a parser dependency is absent. A valid file
                # with no extractable symbols may still use the module fallback.
                tree_sitter_parser(source_file.language)
            chunks = list(self.tree_sitter_symbol_chunks(source_file))
            if chunks:
                for chunk in chunks:
                    for bounded in self.bound_chunk(
                        chunk,
                        max_chunk_tokens=max_chunk_tokens,
                        overlap_tokens=overlap_tokens,
                    ):
                        yield self.chunk_payload(organization_id, project_id, file_id, source_file, **bounded)
                return
        for chunk in self.line_chunks(source_file, max_chunk_lines=max_chunk_lines):
            for bounded in self.bound_chunk(chunk, max_chunk_tokens=max_chunk_tokens, overlap_tokens=overlap_tokens):
                yield self.chunk_payload(organization_id, project_id, file_id, source_file, **bounded)

    def bound_chunk(
        self,
        chunk: dict[str, Any],
        *,
        max_chunk_tokens: int,
        overlap_tokens: int,
    ) -> Iterable[dict[str, Any]]:
        content = str(chunk["content"])
        if estimate_tokens(content) <= max_chunk_tokens:
            yield chunk
            return
        lines = content.splitlines()
        max_chars = max_chunk_tokens * 4
        overlap_chars = overlap_tokens * 4
        start = 0
        part = 0
        while start < len(content):
            end = min(len(content), start + max_chars)
            if end < len(content):
                newline = content.rfind("\n", start, end)
                if newline > start:
                    end = newline
            segment = content[start:end]
            start_line_offset = content[:start].count("\n")
            end_line_offset = start_line_offset + max(0, segment.count("\n"))
            bounded = dict(chunk)
            bounded["content"] = segment
            bounded["start_line"] = int(chunk["start_line"]) + start_line_offset
            bounded["end_line"] = min(int(chunk["end_line"]), int(chunk["start_line"]) + end_line_offset)
            bounded["kind"] = f"{chunk['kind']}_part"
            bounded["metadata"] = {**dict(chunk.get("metadata") or {}), "parent_symbol": chunk.get("symbol_name"), "part": part}
            yield bounded
            part += 1
            if end >= len(content):
                break
            start = max(start + 1, end - overlap_chars)

    def python_symbol_chunks(self, source_file: SourceFile) -> Iterable[dict[str, Any]]:
        try:
            tree = ast.parse(source_file.content)
        except SyntaxError:
            return []
        lines = source_file.content.splitlines()
        facts = self.python_module_facts(tree)
        chunks: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = max(1, getattr(node, "lineno", 1))
            end = max(start, getattr(node, "end_lineno", start))
            content = "\n".join(lines[start - 1 : end])
            signature = self.python_signature(node)
            calls = self.python_calls(node)
            mutations = self.python_mutations(node)
            decorators = [
                self.safe_unparse(decorator)
                for decorator in getattr(node, "decorator_list", [])
                if self.safe_unparse(decorator)
            ]
            chunks.append(
                {
                    "start_line": start,
                    "end_line": end,
                    "symbol_name": node.name,
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "content": content,
                    "metadata": {
                        "parser": "python_ast",
                        "signature": signature,
                        "imports": facts.imports,
                        "calls": calls,
                        "dependencies": sorted(set(facts.imports + calls)),
                        "state_mutations": mutations,
                        "decorators": decorators,
                        "docstring": ast.get_docstring(node) or "",
                    },
                }
            )
        return sorted(chunks, key=lambda item: (item["start_line"], item["end_line"]))

    def tree_sitter_symbol_chunks(self, source_file: SourceFile) -> Iterable[dict[str, Any]]:
        """Extract hierarchy-preserving symbols and local program facts.

        Tree-sitter byte offsets are used rather than decoded character offsets,
        which keeps line and symbol boundaries correct for Unicode source files.
        """

        try:
            parser = tree_sitter_parser(source_file.language)
            source_bytes = source_file.content.encode("utf-8", "surrogatepass")
            try:
                tree = parser.parse(source_bytes)
            except TypeError:
                tree = parser.parse(source_file.content)
        except (RuntimeError, ValueError, TypeError):
            return []
        root = ts_value(getattr(tree, "root_node"))
        if ts_has_error(root) and not any(
            ts_node_type(child) in TREE_SITTER_SYMBOL_TYPES[source_file.language]
            for child in self.walk_tree_sitter(root)
        ):
            return []
        file_imports = self.generic_imports(source_file.language, source_file.content)
        chunks: list[dict[str, Any]] = []
        for node in self.walk_tree_sitter(root):
            node_type = ts_node_type(node)
            kind = TREE_SITTER_SYMBOL_TYPES[source_file.language].get(node_type)
            if kind is None:
                continue
            name = self.tree_sitter_symbol_name(node, source_bytes)
            if not name:
                continue
            content = source_bytes[ts_byte(node, "start") : ts_byte(node, "end")].decode("utf-8", "replace")
            start_line = ts_row(node, "start") + 1
            end_line = max(start_line, ts_row(node, "end") + 1)
            signature = self.tree_sitter_signature(node, source_bytes)
            calls = self.tree_sitter_calls(node, source_bytes)
            mutations = self.tree_sitter_mutations(node, source_bytes)
            parent_symbols = self.tree_sitter_parent_symbols(node, source_bytes)
            qualified_symbol = "::".join([*parent_symbols, name])
            chunks.append(
                {
                    "start_line": start_line,
                    "end_line": end_line,
                    "symbol_name": name,
                    "kind": kind,
                    "content": content,
                    "metadata": {
                        "parser": "tree_sitter",
                        "tree_sitter_language": source_file.language,
                        "tree_sitter_node_type": node_type,
                        "signature": signature,
                        "qualified_symbol": qualified_symbol,
                        "parent_symbols": parent_symbols,
                        "imports": file_imports,
                        "calls": calls,
                        "dependencies": sorted(set(file_imports + calls)),
                        "state_mutations": mutations,
                        "parse_has_error": ts_has_error(root),
                    },
                }
            )
        return sorted(chunks, key=lambda item: (item["start_line"], -item["end_line"]))

    @staticmethod
    def walk_tree_sitter(root: Any) -> Iterator[Any]:
        stack = [root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(ts_children(node)))

    @staticmethod
    def tree_sitter_node_text(node: Any, source_bytes: bytes) -> str:
        return source_bytes[ts_byte(node, "start") : ts_byte(node, "end")].decode("utf-8", "replace").strip()

    def tree_sitter_symbol_name(self, node: Any, source_bytes: bytes) -> str:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            name = self.tree_sitter_node_text(name_node, source_bytes)
            if name:
                return name[:512]
        declarator = node.child_by_field_name("declarator")
        if declarator is not None:
            candidates = [
                child
                for child in self.walk_tree_sitter(declarator)
                if ts_node_type(child) in {"identifier", "field_identifier", "type_identifier"}
            ]
            if candidates:
                return self.tree_sitter_node_text(candidates[-1], source_bytes)[:512]
        for child in self.walk_tree_sitter(node):
            if child is node:
                continue
            if ts_node_type(child) in {"identifier", "field_identifier", "type_identifier"}:
                return self.tree_sitter_node_text(child, source_bytes)[:512]
        return ""

    def tree_sitter_signature(self, node: Any, source_bytes: bytes) -> str:
        body = node.child_by_field_name("body")
        start_byte = ts_byte(node, "start")
        end_byte = ts_byte(body, "start") if body is not None else min(ts_byte(node, "end"), start_byte + 2048)
        signature = source_bytes[start_byte:end_byte].decode("utf-8", "replace").strip()
        return " ".join(signature.split())[:2048]

    def tree_sitter_calls(self, node: Any, source_bytes: bytes) -> list[str]:
        calls: set[str] = set()
        for child in self.walk_tree_sitter(node):
            if child is node or ts_node_type(child) not in TREE_SITTER_CALL_TYPES:
                continue
            function = child.child_by_field_name("function") or child.child_by_field_name("name")
            named_children = ts_children(child, named=True)
            if function is None and named_children:
                function = named_children[0]
            if function is None:
                continue
            value = self.tree_sitter_node_text(function, source_bytes)
            value = re.sub(r"\s+", "", value)
            if value and len(value) <= 512:
                calls.add(value)
        return sorted(calls)

    def tree_sitter_mutations(self, node: Any, source_bytes: bytes) -> list[str]:
        mutations: set[str] = set()
        for child in self.walk_tree_sitter(node):
            if child is node or ts_node_type(child) not in TREE_SITTER_MUTATION_TYPES:
                continue
            target = child.child_by_field_name("left") or child.child_by_field_name("name")
            named_children = ts_children(child, named=True)
            if target is None and named_children:
                target = named_children[0]
            if target is None:
                continue
            value = self.tree_sitter_node_text(target, source_bytes)
            if value and len(value) <= 512:
                mutations.add(value)
        return sorted(mutations)

    def tree_sitter_parent_symbols(self, node: Any, source_bytes: bytes) -> list[str]:
        parents: list[str] = []
        parent = ts_parent(node)
        symbol_types = TREE_SITTER_SYMBOL_TYPES
        while parent is not None:
            if any(ts_node_type(parent) in values for values in symbol_types.values()):
                name = self.tree_sitter_symbol_name(parent, source_bytes)
                if name:
                    parents.append(name)
            parent = ts_parent(parent)
        return list(reversed(parents))

    def resolve_interprocedural_graph(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Resolve repository-local call edges without inventing ambiguous links."""

        by_exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            symbol = str(chunk.get("symbol_name") or "").strip()
            if not symbol:
                continue
            metadata = dict(chunk.get("metadata") or {})
            qualified = str(metadata.get("qualified_symbol") or symbol)
            by_exact[qualified].append(chunk)
            by_leaf[symbol.split(".")[-1].split("::")[-1]].append(chunk)

        reverse_edges: dict[str, set[str]] = defaultdict(set)
        resolved_edge_count = ambiguous_edge_count = unresolved_edge_count = 0
        for chunk in chunks:
            metadata = dict(chunk.get("metadata") or {})
            resolved: list[dict[str, str]] = []
            ambiguous: list[str] = []
            unresolved: list[str] = []
            for call in sorted(set(str(item) for item in metadata.get("calls", []) if item)):
                normalized = call.replace("?.", ".").replace("->", ".")
                leaf = normalized.split(".")[-1].split("::")[-1]
                candidates = by_exact.get(normalized, []) or by_leaf.get(leaf, [])
                if len(candidates) == 1:
                    target = candidates[0]
                    resolved.append(
                        {
                            "call": call,
                            "target_chunk_id": str(target["id"]),
                            "target_path": str(target["path"]),
                            "target_symbol": str(target.get("symbol_name") or ""),
                        }
                    )
                    reverse_edges[str(target["id"])].add(str(chunk["id"]))
                    resolved_edge_count += 1
                elif len(candidates) > 1:
                    ambiguous.append(call)
                    ambiguous_edge_count += 1
                else:
                    unresolved.append(call)
                    unresolved_edge_count += 1
            metadata["resolved_calls"] = resolved
            metadata["ambiguous_calls"] = ambiguous
            metadata["external_or_unresolved_calls"] = unresolved
            chunk["metadata"] = metadata

        for chunk in chunks:
            metadata = dict(chunk.get("metadata") or {})
            metadata["called_by_chunk_ids"] = sorted(reverse_edges.get(str(chunk["id"]), set()))
            chunk["metadata"] = metadata
        return {
            "resolved_edges": resolved_edge_count,
            "ambiguous_edges": ambiguous_edge_count,
            "external_or_unresolved_edges": unresolved_edge_count,
            "symbols": sum(len(values) for values in by_leaf.values()),
        }

    def line_chunks(self, source_file: SourceFile, *, max_chunk_lines: int) -> Iterable[dict[str, Any]]:
        lines = source_file.content.splitlines()
        if not lines:
            return
        file_imports = self.generic_imports(source_file.language, source_file.content)
        for start_index in range(0, len(lines), max_chunk_lines):
            end_index = min(len(lines), start_index + max_chunk_lines)
            content = "\n".join(lines[start_index:end_index])
            symbol_match = SYMBOL_REGEX.search(content)
            yield {
                "start_line": start_index + 1,
                "end_line": end_index,
                "symbol_name": symbol_match.group(1) if symbol_match else None,
                "kind": "module" if start_index == 0 else "chunk",
                "content": content,
                "metadata": {
                    "parser": (
                        "tree_sitter_module_fallback"
                        if source_file.language in TREE_SITTER_LANGUAGES
                        else "line_chunker"
                    ),
                    "imports": file_imports,
                    "dependencies": file_imports,
                    "signature": self.first_symbol_line(content),
                },
            }

    def python_module_facts(self, tree: ast.AST) -> PythonModuleFacts:
        imports: list[str] = []
        calls: list[str] = []
        mutations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                imports.extend(f"{module}.{alias.name}".strip(".") for alias in node.names)
            elif isinstance(node, ast.Call):
                call = self.call_name(node.func)
                if call:
                    calls.append(call)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                mutations.extend(self.assignment_targets(node))
        return PythonModuleFacts(
            imports=sorted(set(imports)),
            module_calls=sorted(set(calls)),
            module_mutations=sorted(set(mutations)),
        )

    def python_signature(self, node: ast.AST) -> str:
        if isinstance(node, ast.ClassDef):
            bases = [self.safe_unparse(base) for base in node.bases if self.safe_unparse(base)]
            return f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            args = self.safe_unparse(node.args)
            returns = f" -> {self.safe_unparse(node.returns)}" if node.returns is not None else ""
            return f"{prefix} {node.name}({args}){returns}"
        return ""

    def python_calls(self, node: ast.AST) -> list[str]:
        calls: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call = self.call_name(child.func)
                if call:
                    calls.add(call)
        return sorted(calls)

    def python_mutations(self, node: ast.AST) -> list[str]:
        mutations: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                mutations.update(self.assignment_targets(child))
            elif isinstance(child, ast.For):
                mutations.update(self.target_names(child.target))
            elif isinstance(child, ast.With):
                for item in child.items:
                    if item.optional_vars is not None:
                        mutations.update(self.target_names(item.optional_vars))
        return sorted(mutations)

    def assignment_targets(self, node: ast.AST) -> list[str]:
        if isinstance(node, ast.Assign):
            return [name for target in node.targets for name in self.target_names(target)]
        if isinstance(node, ast.AnnAssign):
            return self.target_names(node.target)
        if isinstance(node, ast.AugAssign):
            return self.target_names(node.target)
        return []

    def target_names(self, target: ast.AST) -> list[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, ast.Attribute):
            owner = self.safe_unparse(target.value)
            return [f"{owner}.{target.attr}" if owner else target.attr]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [name for item in target.elts for name in self.target_names(item)]
        return []

    def call_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self.call_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        if isinstance(node, ast.Call):
            return self.call_name(node.func)
        return None

    def safe_unparse(self, node: ast.AST | None) -> str:
        if node is None:
            return ""
        try:
            return ast.unparse(node)
        except Exception:
            return ""

    def generic_imports(self, language: str, content: str) -> list[str]:
        pattern = IMPORT_REGEX_BY_LANGUAGE.get(language)
        if pattern is None:
            return []
        imports: set[str] = set()
        for match in pattern.finditer(content):
            for group in match.groups():
                if group:
                    imports.add(group.strip())
        return sorted(imports)

    def first_symbol_line(self, content: str) -> str:
        for line in content.splitlines():
            if SYMBOL_REGEX.match(line):
                return line.strip()
        return ""

    def chunk_payload(
        self,
        organization_id: str,
        project_id: str,
        file_id: str,
        source_file: SourceFile,
        *,
        start_line: int,
        end_line: int,
        symbol_name: str | None,
        kind: str,
        content: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        chunk_hash = sha256_text(content)
        return {
            "id": stable_uuid(
                organization_id, project_id, source_file.relative_path, symbol_name or "",
                start_line, end_line, chunk_hash, CHUNKER_VERSION,
            ),
            "project_id": project_id,
            "file_id": file_id,
            "path": source_file.relative_path,
            "language": source_file.language,
            "start_line": start_line,
            "end_line": end_line,
            "symbol_name": symbol_name,
            "kind": kind,
            "chunk_hash": chunk_hash,
            "token_count": estimate_tokens(content),
            "content": content,
            "metadata": metadata,
        }


class RAGJobStore(Protocol):
    def create_index_job(self, **kwargs: Any) -> dict[str, Any]: ...
    def get_index_job(self, job_id: str, *, organization_id: str | None = None) -> dict[str, Any] | None: ...
    def claim_index_job(self, *, worker_id: str, lease_seconds: int = 60) -> dict[str, Any] | None: ...
    def heartbeat_index_job(self, job_id: str, *, worker_id: str, lease_seconds: int = 60) -> bool: ...
    def index_job_cancel_requested(self, job_id: str) -> bool: ...
    def request_index_job_cancel(self, job_id: str, *, organization_id: str) -> dict[str, Any]: ...
    def complete_index_job(self, job_id: str, **kwargs: Any) -> dict[str, Any]: ...
    def fail_index_job(self, job_id: str, **kwargs: Any) -> dict[str, Any]: ...
    def claim_rag_outbox(self, *, worker_id: str, lease_seconds: int = 60) -> dict[str, Any] | None: ...
    def complete_rag_outbox(self, event_id: str, *, worker_id: str) -> None: ...
    def fail_rag_outbox(self, event_id: str, **kwargs: Any) -> None: ...


class RedisRAGControl:
    """Redis wake-up, cancellation, and distributed-lock transport.

    Postgres/SQLite remains the job source of truth. Redis loss therefore
    delays work but cannot lose or duplicate an index state transition.
    """

    RELEASE_LOCK_LUA = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
      return redis.call('del', KEYS[1])
    end
    return 0
    """
    INDEX_STREAM = "craftly:rag:index-jobs"
    WORKER_GROUP = "craftly-rag-workers"

    def __init__(self, redis_url: str, *, stream_maxlen: int = 100_000) -> None:
        if not redis_url.startswith(("redis://", "rediss://")):
            raise ValueError("RAG control requires redis:// or rediss:// URL")
        self.redis_url = redis_url
        self.stream_maxlen = min(max(stream_maxlen, 1000), 1_000_000)
        self._client: Any = None

    async def client(self) -> Any:
        if self._client is None:
            import redis.asyncio as redis

            self._client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=5,
                health_check_interval=30,
                retry_on_timeout=True,
            )
        return self._client

    async def healthcheck(self) -> None:
        client = await self.client()
        if not await client.ping():
            raise RuntimeError("Redis RAG control ping failed")

    async def notify(self, job: dict[str, Any]) -> None:
        client = await self.client()
        await client.xadd(
            self.INDEX_STREAM,
            {
                "job_id": str(job["id"]),
                "organization_id": str(job["organization_id"]),
                "project_id": str(job["project_id"]),
            },
            maxlen=self.stream_maxlen,
            approximate=True,
        )

    async def wait_for_work(self, worker_id: str, *, timeout_seconds: float) -> bool:
        """Consume one wake signal; Postgres remains the authoritative queue."""

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,127}", worker_id):
            raise ValueError("worker_id contains unsafe characters")
        client = await self.client()
        try:
            await client.xgroup_create(
                self.INDEX_STREAM,
                self.WORKER_GROUP,
                id="$",
                mkstream=True,
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        messages = await client.xreadgroup(
            self.WORKER_GROUP,
            worker_id,
            {self.INDEX_STREAM: ">"},
            count=32,
            block=int(min(max(timeout_seconds, 0.1), 60.0) * 1000),
        )
        message_ids = [message_id for _stream, entries in messages for message_id, _payload in entries]
        if message_ids:
            await client.xack(self.INDEX_STREAM, self.WORKER_GROUP, *message_ids)
        return bool(message_ids)

    async def acquire_project_lock(self, project_id: str, owner: str, *, ttl_seconds: int) -> bool:
        client = await self.client()
        return bool(
            await client.set(
                f"craftly:rag:project-lock:{project_id}",
                owner,
                ex=min(max(ttl_seconds, 30), 3600),
                nx=True,
            )
        )

    async def release_project_lock(self, project_id: str, owner: str) -> None:
        client = await self.client()
        await client.eval(self.RELEASE_LOCK_LUA, 1, f"craftly:rag:project-lock:{project_id}", owner)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class RAGIndexCoordinator:
    """Durable asynchronous repository-index and vector-sync worker."""

    def __init__(
        self,
        store: RAGJobStore | None = None,
        *,
        redis_url: str | None = None,
        production_mode: bool | None = None,
        worker_id: str | None = None,
        lease_seconds: int = 120,
        redis_control: RedisRAGControl | None = None,
    ) -> None:
        self.store = store or LocalStore()
        self.production_mode = (
            production_mode
            if production_mode is not None
            else os.environ.get("CRAFTLY_ENV", "development").lower() == "production"
        )
        active_redis = redis_url or os.environ.get("CRAFTLY_REDIS_URL")
        if self.production_mode and not active_redis:
            raise RuntimeError("production RAG indexing requires CRAFTLY_REDIS_URL")
        self.redis = redis_control or (RedisRAGControl(active_redis) if active_redis else None)
        self._owns_redis = redis_control is None
        self.worker_id = worker_id or f"rag-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self.lease_seconds = min(max(lease_seconds, 30), 900)

    async def submit(
        self,
        *,
        organization_id: str,
        project_id: str,
        idempotency_key: str,
        request: dict[str, Any] | None = None,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        payload = dict(request or {})
        allowed = {"include_suffixes", "max_file_bytes", "max_chunk_lines", "max_chunk_tokens", "overlap_tokens"}
        unexpected = sorted(set(payload) - allowed)
        if unexpected:
            raise ValueError(f"unsupported index request fields: {unexpected}")
        job = await asyncio.to_thread(
            self.store.create_index_job,
            organization_id=organization_id,
            project_id=project_id,
            idempotency_key=idempotency_key,
            request=payload,
            max_attempts=max_attempts,
        )
        if self.redis is not None and job["status"] == "queued":
            await self.redis.notify(job)
        return job

    async def cancel(self, job_id: str, *, organization_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.store.request_index_job_cancel,
            job_id,
            organization_id=organization_id,
        )

    async def run_index_once(self) -> dict[str, Any] | None:
        job = await asyncio.to_thread(
            self.store.claim_index_job,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if job is None:
            return None
        return await self.execute_claimed_index_job(job)

    async def execute_claimed_index_job(self, job: dict[str, Any]) -> dict[str, Any]:
        if job.get("status") != "running" or job.get("lease_owner") != self.worker_id:
            raise ValueError("claimed RAG job is not leased to this worker")
        project_id = str(job["project_id"])
        lock_owner = f"{self.worker_id}:{job['id']}"
        lock_acquired = True
        if self.redis is not None:
            lock_acquired = await self.redis.acquire_project_lock(
                project_id,
                lock_owner,
                ttl_seconds=self.lease_seconds * 2,
            )
        if not lock_acquired:
            return await asyncio.to_thread(
                self.store.fail_index_job,
                str(job["id"]),
                worker_id=self.worker_id,
                error="project index lock is held by another worker",
                retry_delay_seconds=2.0,
                permanent=False,
            )

        stop_heartbeat = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(str(job["id"]), stop_heartbeat))
        try:
            request = dict(job.get("request") or {})
            if request.get("include_suffixes") is not None:
                request["include_suffixes"] = set(str(item) for item in request["include_suffixes"])
            else:
                request.pop("include_suffixes", None)
            report = await asyncio.to_thread(
                RepositoryIndexer(self.store, production_mode=self.production_mode).index_project,
                project_id=project_id,
                organization_id=str(job["organization_id"]),
                cancellation_check=lambda: self.store.index_job_cancel_requested(str(job["id"])),
                **request,
            )
            return await asyncio.to_thread(
                self.store.complete_index_job,
                str(job["id"]),
                worker_id=self.worker_id,
                revision_id=report.revision_id,
                result=report.model_dump(mode="json"),
            )
        except Exception as exc:
            permanent = isinstance(exc, (ValueError, KeyError, PermissionError, FileNotFoundError))
            attempt = int(job.get("attempt") or 1)
            failed = await asyncio.to_thread(
                self.store.fail_index_job,
                str(job["id"]),
                worker_id=self.worker_id,
                error=f"{type(exc).__name__}: {exc}",
                retry_delay_seconds=min(300.0, float(2 ** min(attempt, 8))),
                permanent=permanent,
            )
            if self.redis is not None and failed["status"] == "queued":
                await self.redis.notify(failed)
            return failed
        finally:
            stop_heartbeat.set()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            if self.redis is not None:
                await self.redis.release_project_lock(project_id, lock_owner)

    async def run_vector_sync_once(self) -> dict[str, Any] | None:
        event = await asyncio.to_thread(
            self.store.claim_rag_outbox,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if event is None:
            return None
        return await self.execute_claimed_outbox(event)

    async def execute_claimed_outbox(self, event: dict[str, Any]) -> dict[str, Any]:
        if event.get("status") != "delivering" or event.get("lease_owner") != self.worker_id:
            raise ValueError("claimed RAG outbox event is not leased to this worker")
        try:
            from src.craftly.indexing.vector_index import VectorBackendConfig, create_vector_index

            config = VectorBackendConfig(
                backend="qdrant" if self.production_mode else os.environ.get("CRAFTLY_VECTOR_BACKEND", "local_hashing"),
                dims=int(os.environ.get("CRAFTLY_EMBEDDING_DIMS", "768" if self.production_mode else "384")),
                qdrant_url=os.environ.get("CRAFTLY_QDRANT_URL"),
                embedding_url=os.environ.get("CRAFTLY_EMBEDDING_URL"),
                embedding_manifest_path=os.environ.get("CRAFTLY_EMBEDDING_MANIFEST"),
                production_mode=self.production_mode,
            )
            index = create_vector_index(self.store, config)
            report = await asyncio.to_thread(
                index.sync_project,
                organization_id=str(event["organization_id"]),
                project_id=str(event["project_id"]),
                revision_id=str(event["revision_id"]),
                batch_size=64,
            )
            await asyncio.to_thread(
                self.store.complete_rag_outbox,
                str(event["id"]),
                worker_id=self.worker_id,
            )
            return report.model_dump(mode="json")
        except Exception as exc:
            await asyncio.to_thread(
                self.store.fail_rag_outbox,
                str(event["id"]),
                worker_id=self.worker_id,
                error=f"{type(exc).__name__}: {exc}",
                retry_delay_seconds=min(300.0, float(2 ** min(int(event.get("attempt") or 1), 8))),
            )
            return {"status": "retry_or_dead_letter", "event_id": event["id"], "error_type": type(exc).__name__}

    async def _heartbeat(self, job_id: str, stopped: asyncio.Event) -> None:
        interval = max(5.0, self.lease_seconds / 3)
        while not stopped.is_set():
            try:
                await asyncio.wait_for(stopped.wait(), timeout=interval)
            except asyncio.TimeoutError:
                alive = await asyncio.to_thread(
                    self.store.heartbeat_index_job,
                    job_id,
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                )
                if not alive:
                    return

    async def close(self) -> None:
        if self.redis is not None and self._owns_redis:
            await self.redis.close()


async def _run_worker(args: argparse.Namespace) -> int:
    if args.production and not args.organization_id:
        return await _run_distributed_worker(args)
    store: RAGJobStore
    if args.production:
        dsn = os.environ.get("CRAFTLY_DATABASE_URL", "")
        if not dsn:
            raise RuntimeError("production RAG worker requires CRAFTLY_DATABASE_URL")
        store = PostgresRAGStore(dsn, organization_id=args.organization_id)
    else:
        store = LocalStore(args.sqlite_path) if args.sqlite_path else LocalStore()
    coordinator = RAGIndexCoordinator(
        store,
        redis_url=args.redis_url,
        production_mode=args.production,
        worker_id=args.worker_id,
        lease_seconds=args.lease_seconds,
    )
    try:
        if coordinator.redis is not None:
            await coordinator.redis.healthcheck()
        while True:
            index_result = await coordinator.run_index_once()
            vector_result = await coordinator.run_vector_sync_once()
            if args.once:
                print(json.dumps({"index": index_result, "vector_sync": vector_result}, indent=2, sort_keys=True))
                return 0
            if index_result is None and vector_result is None:
                if coordinator.redis is None:
                    await asyncio.sleep(args.poll_seconds)
                else:
                    try:
                        await coordinator.redis.wait_for_work(coordinator.worker_id, timeout_seconds=args.poll_seconds)
                    except Exception as exc:
                        print(
                            json.dumps(
                                {"event": "rag_redis_wake_failed", "error_type": type(exc).__name__},
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        await asyncio.sleep(args.poll_seconds)
    finally:
        await coordinator.close()
        close = getattr(store, "close", None)
        if callable(close):
            close()


async def _run_distributed_worker(args: argparse.Namespace) -> int:
    dsn = os.environ.get("CRAFTLY_DATABASE_URL", "")
    redis_url = args.redis_url or os.environ.get("CRAFTLY_REDIS_URL", "")
    if not dsn or not redis_url:
        raise RuntimeError("distributed RAG worker requires CRAFTLY_DATABASE_URL and CRAFTLY_REDIS_URL")
    worker_id = args.worker_id or f"rag-dispatch-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    dispatcher = PostgresRAGDispatcher(dsn)
    store_factory = PostgresRAGStoreFactory(dsn, min_pool_size=1, max_pool_size=8)
    control = RedisRAGControl(redis_url)
    await control.healthcheck()
    try:
        while True:
            activity = False
            claimed_job = await asyncio.to_thread(
                dispatcher.claim_index_job,
                worker_id=worker_id,
                lease_seconds=args.lease_seconds,
            )
            if claimed_job is not None:
                activity = True
                store = store_factory.for_organization(str(claimed_job["organization_id"]))
                coordinator = RAGIndexCoordinator(
                    store,
                    production_mode=True,
                    worker_id=worker_id,
                    lease_seconds=args.lease_seconds,
                    redis_control=control,
                )
                try:
                    await coordinator.execute_claimed_index_job(claimed_job)
                finally:
                    await coordinator.close()
            claimed_event = await asyncio.to_thread(
                dispatcher.claim_outbox,
                worker_id=worker_id,
                lease_seconds=args.lease_seconds,
            )
            if claimed_event is not None:
                activity = True
                store = store_factory.for_organization(str(claimed_event["organization_id"]))
                coordinator = RAGIndexCoordinator(
                    store,
                    production_mode=True,
                    worker_id=worker_id,
                    lease_seconds=args.lease_seconds,
                    redis_control=control,
                )
                try:
                    await coordinator.execute_claimed_outbox(claimed_event)
                finally:
                    await coordinator.close()
            if args.once:
                print(
                    json.dumps(
                        {"index": claimed_job, "vector_sync": claimed_event, "worker_id": worker_id},
                        indent=2,
                        sort_keys=True,
                        default=str,
                    )
                )
                return 0
            if not activity:
                try:
                    await control.wait_for_work(worker_id, timeout_seconds=args.poll_seconds)
                except Exception as exc:
                    print(
                        json.dumps(
                            {"event": "rag_redis_wake_failed", "error_type": type(exc).__name__},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    await asyncio.sleep(args.poll_seconds)
    finally:
        await control.close()
        store_factory.close()
        dispatcher.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the durable Craftly Hybrid RAG indexing worker")
    parser.add_argument("worker", nargs="?", default="worker", choices=["worker"])
    parser.add_argument("--redis-url", default=os.environ.get("CRAFTLY_REDIS_URL"))
    parser.add_argument("--worker-id")
    parser.add_argument("--organization-id")
    parser.add_argument("--sqlite-path")
    parser.add_argument("--lease-seconds", type=int, default=120)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 30 <= args.lease_seconds <= 900:
        parser.error("--lease-seconds must be between 30 and 900")
    if not 0.1 <= args.poll_seconds <= 60.0:
        parser.error("--poll-seconds must be between 0.1 and 60")
    raise SystemExit(asyncio.run(_run_worker(args)))


if __name__ == "__main__":
    main()

