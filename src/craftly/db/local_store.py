"""SQLite-backed local store for the Craftly MVP.

The production contract is Postgres. This local store mirrors the MVP tables so
the gateway, indexer, context builder, and tests can run immediately on a
developer machine without external services.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SQLITE_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS organizations (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS organization_members (
  organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  role TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY(organization_id, user_id)
);

CREATE TABLE IF NOT EXISTS project_members (
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,
  role TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY(project_id, user_id)
);

CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  organization_id TEXT NOT NULL DEFAULT 'local' REFERENCES organizations(id),
  name TEXT NOT NULL,
  repo_path TEXT NOT NULL,
  default_branch TEXT NOT NULL DEFAULT 'main',
  index_status TEXT NOT NULL DEFAULT 'not_indexed',
  active_index_revision TEXT,
  index_error TEXT,
  last_indexed_at REAL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS workspace_files (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  language TEXT,
  content_hash TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  index_revision TEXT,
  indexed_at REAL NOT NULL,
  UNIQUE(project_id, path)
);

CREATE TABLE IF NOT EXISTS code_chunks (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  file_id TEXT NOT NULL REFERENCES workspace_files(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  language TEXT,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  symbol_name TEXT,
  kind TEXT NOT NULL,
  chunk_hash TEXT NOT NULL,
  token_count INTEGER NOT NULL,
  content TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  index_revision TEXT,
  indexed_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
  prompt TEXT NOT NULL,
  mode TEXT NOT NULL,
  status TEXT NOT NULL,
  model_profile TEXT NOT NULL,
  confidence REAL,
  summary TEXT,
  started_at REAL,
  finished_at REAL,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS task_graphs (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  run_id TEXT REFERENCES runs(id) ON DELETE CASCADE,
  goal TEXT NOT NULL,
  status TEXT NOT NULL,
  graph_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  task_graph_id TEXT NOT NULL REFERENCES task_graphs(id) ON DELETE CASCADE,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  status TEXT NOT NULL,
  depends_on_json TEXT NOT NULL DEFAULT '[]',
  input_json TEXT NOT NULL DEFAULT '{}',
  output_json TEXT NOT NULL DEFAULT '{}',
  attempt INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 2,
  lease_owner TEXT,
  lease_expires_at REAL,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  started_at REAL,
  finished_at REAL,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS patches (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
  status TEXT NOT NULL,
  diff TEXT NOT NULL,
  files_changed_json TEXT NOT NULL DEFAULT '[]',
  backup_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL,
  applied_at REAL,
  rolled_back_at REAL
);

CREATE TABLE IF NOT EXISTS evaluations (
  id TEXT PRIMARY KEY,
  benchmark TEXT NOT NULL,
  model_profile TEXT NOT NULL,
  status TEXT NOT NULL,
  total INTEGER NOT NULL DEFAULT 0,
  resolved INTEGER NOT NULL DEFAULT 0,
  score REAL,
  report_path TEXT,
  result_json TEXT NOT NULL DEFAULT '{}',
  started_at REAL,
  finished_at REAL,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_entries (
  id TEXT PRIMARY KEY,
  organization_id TEXT NOT NULL DEFAULT 'local' REFERENCES organizations(id),
  project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  content TEXT NOT NULL,
  source_run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
  relevance REAL NOT NULL DEFAULT 0.5,
  success_rate REAL NOT NULL DEFAULT 0.5,
  usage_count INTEGER NOT NULL DEFAULT 0,
  last_used_at REAL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS learning_candidates (
  id TEXT PRIMARY KEY,
  project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
  run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
  patch_id TEXT REFERENCES patches(id) ON DELETE SET NULL,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  prompt TEXT NOT NULL,
  chosen TEXT NOT NULL,
  verification_json TEXT NOT NULL DEFAULT '{}',
  score REAL,
  exported_at REAL,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_messages (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  task_graph_id TEXT NOT NULL REFERENCES task_graphs(id) ON DELETE CASCADE,
  task_id TEXT REFERENCES tasks(id) ON DELETE SET NULL,
  correlation_id TEXT NOT NULL,
  sender_role TEXT NOT NULL CHECK (sender_role IN ('architect', 'coder', 'tester', 'security_reviewer', 'critic', 'verifier', 'orchestrator')),
  recipient_role TEXT NOT NULL CHECK (recipient_role IN ('architect', 'coder', 'tester', 'security_reviewer', 'critic', 'verifier', 'orchestrator', 'broadcast')),
  kind TEXT NOT NULL CHECK (kind IN ('proposal', 'evidence', 'challenge', 'review', 'decision')),
  payload_json TEXT NOT NULL,
  evidence_refs_json TEXT NOT NULL DEFAULT '[]',
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS blackboard_entries (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  task_graph_id TEXT NOT NULL REFERENCES task_graphs(id) ON DELETE CASCADE,
  entry_key TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('fact', 'artifact', 'decision', 'question', 'evidence')),
  value_json TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  immutable INTEGER NOT NULL DEFAULT 0,
  verified INTEGER NOT NULL DEFAULT 0,
  source_message_id TEXT REFERENCES agent_messages(id) ON DELETE SET NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  UNIQUE(run_id, entry_key),
  CHECK (kind <> 'evidence' OR immutable = 1)
);

CREATE TABLE IF NOT EXISTS failure_records (
  id TEXT PRIMARY KEY,
  project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
  run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
  task_id TEXT REFERENCES tasks(id) ON DELETE SET NULL,
  signature TEXT NOT NULL,
  cluster_key TEXT NOT NULL,
  raw_error TEXT NOT NULL,
  root_cause TEXT,
  patch_id TEXT REFERENCES patches(id) ON DELETE SET NULL,
  verification_ref TEXT,
  status TEXT NOT NULL DEFAULT 'observed',
  occurrence_count INTEGER NOT NULL DEFAULT 1,
  dataset_candidate_id TEXT REFERENCES learning_candidates(id) ON DELETE SET NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  first_seen_at REAL NOT NULL,
  last_seen_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rag_index_revisions (
  id TEXT PRIMARY KEY,
  organization_id TEXT NOT NULL REFERENCES organizations(id),
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  source_revision TEXT NOT NULL,
  source_snapshot_sha256 TEXT NOT NULL,
  chunker_version TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('building', 'committed', 'failed', 'superseded')),
  manifest_json TEXT NOT NULL DEFAULT '{}',
  error TEXT,
  created_at REAL NOT NULL,
  committed_at REAL
);

CREATE TABLE IF NOT EXISTS rag_index_jobs (
  id TEXT PRIMARY KEY,
  organization_id TEXT NOT NULL REFERENCES organizations(id),
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  revision_id TEXT REFERENCES rag_index_revisions(id) ON DELETE SET NULL,
  idempotency_key TEXT NOT NULL,
  status TEXT NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  request_json TEXT NOT NULL DEFAULT '{}',
  result_json TEXT NOT NULL DEFAULT '{}',
  lease_owner TEXT,
  lease_expires_at REAL,
  available_at REAL NOT NULL DEFAULT 0,
  error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  finished_at REAL,
  UNIQUE(organization_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS rag_outbox_events (
  id TEXT PRIMARY KEY,
  organization_id TEXT NOT NULL REFERENCES organizations(id),
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  revision_id TEXT NOT NULL REFERENCES rag_index_revisions(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  attempt INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 5,
  lease_owner TEXT,
  lease_expires_at REAL,
  error TEXT,
  available_at REAL NOT NULL,
  created_at REAL NOT NULL,
  delivered_at REAL
);

INSERT OR IGNORE INTO organizations(id, name, status, created_at)
VALUES ('local', 'Local Development Organization', 'active', 0);

CREATE INDEX IF NOT EXISTS idx_workspace_files_project_path ON workspace_files(project_id, path);
CREATE INDEX IF NOT EXISTS idx_code_chunks_project_path ON code_chunks(project_id, path);
CREATE INDEX IF NOT EXISTS idx_code_chunks_project_symbol ON code_chunks(project_id, symbol_name);
CREATE INDEX IF NOT EXISTS idx_runs_project_status ON runs(project_id, status);
CREATE INDEX IF NOT EXISTS idx_tasks_graph_status ON tasks(task_graph_id, status);
CREATE INDEX IF NOT EXISTS idx_memory_project_kind ON memory_entries(project_id, kind);
CREATE INDEX IF NOT EXISTS idx_messages_run_created ON agent_messages(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_correlation ON agent_messages(correlation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_blackboard_run_kind ON blackboard_entries(run_id, kind);
CREATE INDEX IF NOT EXISTS idx_failures_cluster ON failure_records(project_id, cluster_key);
CREATE INDEX IF NOT EXISTS idx_rag_revisions_project ON rag_index_revisions(project_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_one_building_revision ON rag_index_revisions(project_id) WHERE status = 'building';
CREATE INDEX IF NOT EXISTS idx_rag_jobs_status ON rag_index_jobs(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_rag_outbox_pending ON rag_outbox_events(status, available_at);
"""


def default_store_path() -> Path:
    return Path(os.environ.get("CRAFTLY_SQLITE_PATH", "artifacts/craftly/craftly.sqlite3"))


def now_unix() -> float:
    return time.time()


def new_id() -> str:
    return str(uuid.uuid4())


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


class LocalStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    @property
    def connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._connection is None:
                self._connection = sqlite3.connect(self.path, check_same_thread=False)
                self._connection.row_factory = sqlite3.Row
                self._connection.execute("PRAGMA foreign_keys = ON")
                self._connection.executescript(SQLITE_SCHEMA)
                self._ensure_runtime_columns(self._connection)
                self._connection.commit()
            return self._connection

    def _ensure_runtime_columns(self, connection: sqlite3.Connection) -> None:
        task_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
        }
        if "attempt" not in task_columns:
            connection.execute("ALTER TABLE tasks ADD COLUMN attempt INTEGER NOT NULL DEFAULT 0")
        if "max_attempts" not in task_columns:
            connection.execute(
                "ALTER TABLE tasks ADD COLUMN max_attempts INTEGER NOT NULL DEFAULT 2"
            )
        if "lease_owner" not in task_columns:
            connection.execute("ALTER TABLE tasks ADD COLUMN lease_owner TEXT")
        if "lease_expires_at" not in task_columns:
            connection.execute("ALTER TABLE tasks ADD COLUMN lease_expires_at REAL")
        if "cancel_requested" not in task_columns:
            connection.execute(
                "ALTER TABLE tasks ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
            )
        project_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(projects)").fetchall()
        }
        if "organization_id" not in project_columns:
            connection.execute(
                "ALTER TABLE projects ADD COLUMN organization_id TEXT NOT NULL DEFAULT 'local'"
            )
        if "active_index_revision" not in project_columns:
            connection.execute("ALTER TABLE projects ADD COLUMN active_index_revision TEXT")
        if "index_error" not in project_columns:
            connection.execute("ALTER TABLE projects ADD COLUMN index_error TEXT")
        memory_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(memory_entries)").fetchall()
        }
        if "organization_id" not in memory_columns:
            connection.execute(
                "ALTER TABLE memory_entries ADD COLUMN organization_id TEXT NOT NULL DEFAULT 'local'"
            )
        file_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(workspace_files)").fetchall()
        }
        if "index_revision" not in file_columns:
            connection.execute("ALTER TABLE workspace_files ADD COLUMN index_revision TEXT")
        chunk_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(code_chunks)").fetchall()
        }
        if "index_revision" not in chunk_columns:
            connection.execute("ALTER TABLE code_chunks ADD COLUMN index_revision TEXT")
        job_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(rag_index_jobs)").fetchall()
        }
        for column, declaration in {
            "request_json": "TEXT NOT NULL DEFAULT '{}'",
            "result_json": "TEXT NOT NULL DEFAULT '{}'",
            "lease_owner": "TEXT",
            "lease_expires_at": "REAL",
            "available_at": "REAL NOT NULL DEFAULT 0",
            "finished_at": "REAL",
        }.items():
            if column not in job_columns:
                connection.execute(f"ALTER TABLE rag_index_jobs ADD COLUMN {column} {declaration}")
        outbox_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(rag_outbox_events)").fetchall()
        }
        for column, declaration in {
            "max_attempts": "INTEGER NOT NULL DEFAULT 5",
            "lease_owner": "TEXT",
            "lease_expires_at": "REAL",
            "error": "TEXT",
        }.items():
            if column not in outbox_columns:
                connection.execute(
                    f"ALTER TABLE rag_outbox_events ADD COLUMN {column} {declaration}"
                )

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> LocalStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def ensure_organization(
        self, organization_id: str, *, name: str | None = None
    ) -> dict[str, Any]:
        if not organization_id or len(organization_id) > 128:
            raise ValueError("organization_id must contain 1-128 characters")
        self.connection.execute(
            "INSERT OR IGNORE INTO organizations(id, name, status, created_at) VALUES (?, ?, 'active', ?)",
            (organization_id, name or organization_id, now_unix()),
        )
        self.connection.commit()
        row = self.connection.execute(
            "SELECT * FROM organizations WHERE id = ?", (organization_id,)
        ).fetchone()
        result = row_to_dict(row)
        if result is None:
            raise RuntimeError("organization insert did not round-trip")
        return result

    def add_organization_member(
        self, organization_id: str, user_id: str, *, role: str = "member"
    ) -> None:
        if role not in {"owner", "admin", "member", "viewer"}:
            raise ValueError("unsupported organization role")
        self.ensure_organization(organization_id)
        self.connection.execute(
            """
            INSERT INTO organization_members(organization_id, user_id, role, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(organization_id, user_id) DO UPDATE SET role = excluded.role
            """,
            (organization_id, user_id, role, now_unix()),
        )
        self.connection.commit()

    def create_project(
        self,
        *,
        name: str,
        repo_path: str,
        default_branch: str = "main",
        organization_id: str = "local",
        owner_user_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        project_id = project_id or new_id()
        timestamp = now_unix()
        self.ensure_organization(organization_id)
        if owner_user_id:
            self.add_organization_member(organization_id, owner_user_id, role="owner")
        self.connection.execute(
            """
            INSERT INTO projects(id, organization_id, name, repo_path, default_branch, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                organization_id,
                name,
                str(Path(repo_path).resolve()),
                default_branch,
                timestamp,
                timestamp,
            ),
        )
        self.connection.commit()
        if owner_user_id:
            self.connection.execute(
                "INSERT INTO project_members(project_id, organization_id, user_id, role, created_at) VALUES (?, ?, ?, 'owner', ?)",
                (project_id, organization_id, owner_user_id, timestamp),
            )
            self.connection.commit()
        project = self.get_project(project_id)
        if project is None:
            raise RuntimeError("project insert did not round-trip")
        return project

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        return row_to_dict(row)

    def require_project_access(
        self,
        project_id: str,
        organization_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM projects WHERE id = ? AND organization_id = ?",
            (project_id, organization_id),
        ).fetchone()
        project = row_to_dict(row)
        if project is None:
            raise KeyError("project not found in organization")
        if user_id is not None:
            membership = self.connection.execute(
                "SELECT 1 FROM project_members WHERE project_id = ? AND organization_id = ? AND user_id = ?",
                (project_id, organization_id, user_id),
            ).fetchone()
            if membership is None:
                raise PermissionError("user is not a member of this project")
        return project

    def delete_project(self, project_id: str) -> bool:
        cursor = self.connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        self.connection.commit()
        return cursor.rowcount == 1

    def update_project_index_status(
        self, project_id: str, status: str, *, indexed_at: float | None = None
    ) -> None:
        self.connection.execute(
            """
            UPDATE projects
            SET index_status = ?, last_indexed_at = COALESCE(?, last_indexed_at), updated_at = ?
            WHERE id = ?
            """,
            (status, indexed_at, now_unix(), project_id),
        )
        self.connection.commit()

    def create_session(self, *, project_id: str, title: str) -> dict[str, Any]:
        if self.get_project(project_id) is None:
            raise KeyError(f"unknown project: {project_id}")
        session_id = new_id()
        timestamp = now_unix()
        self.connection.execute(
            """
            INSERT INTO sessions(id, project_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, project_id, title, timestamp, timestamp),
        )
        self.connection.commit()
        row = self.connection.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        result = row_to_dict(row)
        if result is None:
            raise RuntimeError("session insert did not round-trip")
        return result

    def create_run(
        self,
        *,
        project_id: str,
        session_id: str | None,
        prompt: str,
        mode: str,
        model_profile: str,
        status: str = "queued",
    ) -> dict[str, Any]:
        if self.get_project(project_id) is None:
            raise KeyError(f"unknown project: {project_id}")
        run_id = new_id()
        timestamp = now_unix()
        self.connection.execute(
            """
            INSERT INTO runs(id, project_id, session_id, prompt, mode, status, model_profile, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, project_id, session_id, prompt, mode, status, model_profile, timestamp),
        )
        self.connection.commit()
        row = self.connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        result = row_to_dict(row)
        if result is None:
            raise RuntimeError("run insert did not round-trip")
        return result

    def update_run_status(
        self,
        run_id: str,
        status: str,
        *,
        summary: str | None = None,
        confidence: float | None = None,
        finished_at: float | None = None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE runs
            SET status = ?,
                summary = COALESCE(?, summary),
                confidence = COALESCE(?, confidence),
                finished_at = COALESCE(?, finished_at)
            WHERE id = ?
            """,
            (status, summary, confidence, finished_at, run_id),
        )
        self.connection.commit()

    def create_task_graph(
        self,
        *,
        project_id: str,
        run_id: str,
        goal: str,
        status: str,
        graph: dict[str, Any],
    ) -> dict[str, Any]:
        graph_id = str(graph["task_graph_id"])
        timestamp = now_unix()
        self.connection.execute(
            """
            INSERT INTO task_graphs(id, project_id, run_id, goal, status, graph_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                graph_id,
                project_id,
                run_id,
                goal,
                status,
                json.dumps(graph, sort_keys=True),
                timestamp,
                timestamp,
            ),
        )
        for node in graph.get("nodes", []):
            self.connection.execute(
                """
                INSERT INTO tasks(
                  id, task_graph_id, run_id, kind, title, status, depends_on_json,
                  input_json, output_json, attempt, max_attempts, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node["node_id"],
                    graph_id,
                    run_id,
                    node["kind"],
                    node["title"],
                    node["status"],
                    json.dumps(node.get("depends_on", []), sort_keys=True),
                    json.dumps(node.get("inputs", {}), sort_keys=True),
                    json.dumps(node.get("outputs", {}), sort_keys=True),
                    int(node.get("attempt", 0)),
                    int(node.get("max_attempts", 2)),
                    timestamp,
                ),
            )
        self.connection.commit()
        result = self.get_task_graph(graph_id)
        if result is None:
            raise RuntimeError("task graph insert did not round-trip")
        return result

    def get_task_graph(self, task_graph_id: str) -> dict[str, Any] | None:
        graph_row = self.connection.execute(
            "SELECT * FROM task_graphs WHERE id = ?", (task_graph_id,)
        ).fetchone()
        graph = row_to_dict(graph_row)
        if graph is None:
            return None
        graph_payload = json.loads(graph["graph_json"])
        graph_payload["status"] = graph["status"]
        graph_payload["updated_at_unix"] = graph["updated_at"]
        task_rows = self.connection.execute(
            "SELECT * FROM tasks WHERE task_graph_id = ? ORDER BY created_at, rowid",
            (task_graph_id,),
        ).fetchall()
        graph_payload["nodes"] = [
            {
                "node_id": row["id"],
                "kind": row["kind"],
                "title": row["title"],
                "status": row["status"],
                "depends_on": json.loads(row["depends_on_json"]),
                "inputs": json.loads(row["input_json"]),
                "outputs": json.loads(row["output_json"]),
                "attempt": int(row["attempt"] or 0),
                "max_attempts": int(row["max_attempts"] or 2),
                "error": row["error"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
            }
            for row in task_rows
        ]
        return graph_payload

    def update_task_graph_status(self, task_graph_id: str, status: str) -> None:
        self.connection.execute(
            """
            UPDATE task_graphs
            SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, now_unix(), task_graph_id),
        )
        self.connection.commit()

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        data = row_to_dict(row)
        if data is None:
            return None
        data["depends_on"] = json.loads(data.pop("depends_on_json") or "[]")
        data["inputs"] = json.loads(data.pop("input_json") or "{}")
        data["outputs"] = json.loads(data.pop("output_json") or "{}")
        data["attempt"] = int(data.get("attempt") or 0)
        data["max_attempts"] = int(data.get("max_attempts") or 2)
        return data

    def list_tasks(self, task_graph_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM tasks WHERE task_graph_id = ? ORDER BY created_at, rowid",
            (task_graph_id,),
        ).fetchall()
        tasks: list[dict[str, Any]] = []
        for row in rows:
            data = row_to_dict(row) or {}
            data["depends_on"] = json.loads(data.pop("depends_on_json") or "[]")
            data["inputs"] = json.loads(data.pop("input_json") or "{}")
            data["outputs"] = json.loads(data.pop("output_json") or "{}")
            data["attempt"] = int(data.get("attempt") or 0)
            data["max_attempts"] = int(data.get("max_attempts") or 2)
            tasks.append(data)
        return tasks

    def update_task_state(
        self,
        task_id: str,
        *,
        status: str,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> None:
        timestamp = now_unix()
        self.connection.execute(
            """
            UPDATE tasks
            SET status = ?,
                output_json = CASE WHEN ? IS NULL THEN output_json ELSE ? END,
                error = ?,
                started_at = CASE WHEN ? THEN COALESCE(started_at, ?) ELSE started_at END,
                finished_at = CASE WHEN ? THEN ? ELSE finished_at END,
                lease_owner = CASE WHEN ? IN ('completed', 'failed', 'cancelled', 'queued') THEN NULL ELSE lease_owner END,
                lease_expires_at = CASE WHEN ? IN ('completed', 'failed', 'cancelled', 'queued') THEN NULL ELSE lease_expires_at END
            WHERE id = ?
            """,
            (
                status,
                None if outputs is None else 1,
                json.dumps(outputs or {}, sort_keys=True),
                error,
                started,
                timestamp,
                finished,
                timestamp,
                status,
                status,
                task_id,
            ),
        )
        self.connection.commit()

    def update_task_attempt(
        self,
        task_id: str,
        *,
        attempt: int,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            UPDATE tasks
            SET attempt = ?,
                output_json = CASE WHEN ? IS NULL THEN output_json ELSE ? END,
                error = ?
            WHERE id = ?
            """,
            (
                attempt,
                None if outputs is None else 1,
                json.dumps(outputs or {}, sort_keys=True),
                error,
                task_id,
            ),
        )
        self.connection.commit()

    def claim_tasks(
        self,
        task_graph_id: str,
        task_ids: list[str],
        *,
        worker_id: str,
        lease_seconds: float,
    ) -> list[dict[str, Any]]:
        """Atomically claim queued tasks using a bounded worker lease."""

        if not task_ids:
            return []
        if not worker_id or len(worker_id) > 200:
            raise ValueError("worker_id must contain 1-200 characters")
        if not 1.0 <= lease_seconds <= 86_400.0:
            raise ValueError("lease_seconds must be between 1 and 86400")
        now = now_unix()
        claimed: list[str] = []
        with self._lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    UPDATE tasks
                    SET attempt = attempt + 1,
                        status = CASE WHEN attempt + 1 >= max_attempts THEN 'failed' ELSE 'queued' END,
                        error = 'worker lease expired',
                        finished_at = CASE WHEN attempt + 1 >= max_attempts THEN ? ELSE finished_at END,
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE task_graph_id = ? AND status = 'running'
                      AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?
                      AND cancel_requested = 0
                    """,
                    (now, task_graph_id, now),
                )
                for task_id in task_ids:
                    cursor = connection.execute(
                        """
                        UPDATE tasks
                        SET status = 'running', lease_owner = ?, lease_expires_at = ?,
                            started_at = COALESCE(started_at, ?)
                        WHERE id = ? AND task_graph_id = ? AND status = 'queued'
                          AND cancel_requested = 0
                        """,
                        (worker_id, now + lease_seconds, now, task_id, task_graph_id),
                    )
                    if cursor.rowcount == 1:
                        claimed.append(task_id)
                if claimed:
                    connection.execute(
                        "UPDATE task_graphs SET status = 'running', updated_at = ? WHERE id = ?",
                        (now, task_graph_id),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return [task for task_id in claimed if (task := self.get_task(task_id)) is not None]

    def recover_expired_task_leases(self, task_graph_id: str) -> int:
        timestamp = now_unix()
        cursor = self.connection.execute(
            """
            UPDATE tasks
            SET attempt = attempt + 1,
                status = CASE WHEN attempt + 1 >= max_attempts THEN 'failed' ELSE 'queued' END,
                lease_owner = NULL, lease_expires_at = NULL,
                error = 'worker lease expired',
                finished_at = CASE WHEN attempt + 1 >= max_attempts THEN ? ELSE finished_at END
            WHERE task_graph_id = ? AND status = 'running'
              AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?
              AND cancel_requested = 0
            """,
            (timestamp, task_graph_id, timestamp),
        )
        self.connection.commit()
        return cursor.rowcount

    def renew_task_lease(self, task_id: str, *, worker_id: str, lease_seconds: float) -> bool:
        if not 1.0 <= lease_seconds <= 86_400.0:
            raise ValueError("lease_seconds must be between 1 and 86400")
        cursor = self.connection.execute(
            """
            UPDATE tasks SET lease_expires_at = ?
            WHERE id = ? AND status = 'running' AND lease_owner = ? AND cancel_requested = 0
            """,
            (now_unix() + lease_seconds, task_id, worker_id),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def cancel_task_graph(self, task_graph_id: str) -> int:
        timestamp = now_unix()
        with self._lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    UPDATE tasks
                    SET cancel_requested = 1,
                        status = CASE WHEN status IN ('queued', 'running') THEN 'cancelled' ELSE status END,
                        finished_at = CASE WHEN status IN ('queued', 'running') THEN ? ELSE finished_at END,
                        lease_owner = NULL,
                        lease_expires_at = NULL
                    WHERE task_graph_id = ? AND status NOT IN ('completed', 'failed', 'cancelled')
                    """,
                    (timestamp, task_graph_id),
                )
                connection.execute(
                    "UPDATE task_graphs SET status = 'cancelled', updated_at = ? WHERE id = ?",
                    (timestamp, task_graph_id),
                )
                connection.commit()
                return cursor.rowcount
            except Exception:
                connection.rollback()
                raise

    def insert_agent_message(self, message: dict[str, Any]) -> dict[str, Any]:
        timestamp = float(message.get("created_at_unix") or now_unix())
        self.connection.execute(
            """
            INSERT INTO agent_messages(
              id, run_id, task_graph_id, task_id, correlation_id, sender_role,
              recipient_role, kind, payload_json, evidence_refs_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message["message_id"],
                message["run_id"],
                message["task_graph_id"],
                message.get("task_id"),
                message["correlation_id"],
                message["sender_role"],
                message["recipient_role"],
                message["kind"],
                json.dumps(message.get("payload", {}), sort_keys=True),
                json.dumps(message.get("evidence_refs", []), sort_keys=True),
                timestamp,
            ),
        )
        self.connection.commit()
        result = self.get_agent_message(str(message["message_id"]))
        if result is None:
            raise RuntimeError("agent message insert did not round-trip")
        return result

    def get_agent_message(self, message_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM agent_messages WHERE id = ?", (message_id,)
        ).fetchone()
        return self._agent_message_row(row)

    def list_agent_messages(
        self,
        run_id: str,
        *,
        correlation_id: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 10_000:
            raise ValueError("message limit must be between 1 and 10000")
        if correlation_id:
            rows = self.connection.execute(
                """
                SELECT * FROM agent_messages
                WHERE run_id = ? AND correlation_id = ?
                ORDER BY created_at, rowid LIMIT ?
                """,
                (run_id, correlation_id, limit),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM agent_messages WHERE run_id = ? ORDER BY created_at, rowid LIMIT ?",
                (run_id, limit),
            ).fetchall()
        return [item for row in rows if (item := self._agent_message_row(row)) is not None]

    def _agent_message_row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        data = row_to_dict(row)
        if data is None:
            return None
        data["message_id"] = data.pop("id")
        data["payload"] = json.loads(data.pop("payload_json") or "{}")
        data["evidence_refs"] = json.loads(data.pop("evidence_refs_json") or "[]")
        data["created_at_unix"] = data.pop("created_at")
        return data

    def put_blackboard_entry(
        self,
        *,
        entry_id: str,
        run_id: str,
        task_graph_id: str,
        entry_key: str,
        kind: str,
        value: dict[str, Any],
        immutable: bool,
        verified: bool,
        source_message_id: str | None,
        expected_version: int | None,
    ) -> dict[str, Any]:
        timestamp = now_unix()
        with self._lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = connection.execute(
                    "SELECT * FROM blackboard_entries WHERE run_id = ? AND entry_key = ?",
                    (run_id, entry_key),
                ).fetchone()
                if current is None:
                    if expected_version not in {None, 0}:
                        raise RuntimeError("blackboard version conflict: entry does not exist")
                    connection.execute(
                        """
                        INSERT INTO blackboard_entries(
                          id, run_id, task_graph_id, entry_key, kind, value_json, version,
                          immutable, verified, source_message_id, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
                        """,
                        (
                            entry_id,
                            run_id,
                            task_graph_id,
                            entry_key,
                            kind,
                            json.dumps(value, sort_keys=True),
                            int(immutable),
                            int(verified),
                            source_message_id,
                            timestamp,
                            timestamp,
                        ),
                    )
                else:
                    current_version = int(current["version"])
                    if bool(current["immutable"]):
                        raise RuntimeError("immutable blackboard evidence cannot be changed")
                    if expected_version is None or expected_version != current_version:
                        raise RuntimeError(
                            f"blackboard version conflict: expected {expected_version}, current {current_version}"
                        )
                    connection.execute(
                        """
                        UPDATE blackboard_entries
                        SET kind = ?, value_json = ?, version = version + 1,
                            verified = ?, source_message_id = ?, updated_at = ?
                        WHERE run_id = ? AND entry_key = ? AND version = ?
                        """,
                        (
                            kind,
                            json.dumps(value, sort_keys=True),
                            int(verified),
                            source_message_id,
                            timestamp,
                            run_id,
                            entry_key,
                            current_version,
                        ),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        result = self.get_blackboard_entry(run_id, entry_key)
        if result is None:
            raise RuntimeError("blackboard write did not round-trip")
        return result

    def get_blackboard_entry(self, run_id: str, entry_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM blackboard_entries WHERE run_id = ? AND entry_key = ?",
            (run_id, entry_key),
        ).fetchone()
        return self._blackboard_row(row)

    def list_blackboard_entries(
        self, run_id: str, *, kind: str | None = None
    ) -> list[dict[str, Any]]:
        if kind:
            rows = self.connection.execute(
                "SELECT * FROM blackboard_entries WHERE run_id = ? AND kind = ? ORDER BY created_at, rowid",
                (run_id, kind),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM blackboard_entries WHERE run_id = ? ORDER BY created_at, rowid",
                (run_id,),
            ).fetchall()
        return [item for row in rows if (item := self._blackboard_row(row)) is not None]

    def _blackboard_row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        data = row_to_dict(row)
        if data is None:
            return None
        data["entry_id"] = data.pop("id")
        data["value"] = json.loads(data.pop("value_json") or "{}")
        data["immutable"] = bool(data["immutable"])
        data["verified"] = bool(data["verified"])
        data["created_at_unix"] = data.pop("created_at")
        data["updated_at_unix"] = data.pop("updated_at")
        return data

    def record_failure(
        self,
        *,
        project_id: str | None,
        run_id: str | None,
        task_id: str | None,
        signature: str,
        cluster_key: str,
        raw_error: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timestamp = now_unix()
        with self._lock:
            row = self.connection.execute(
                """
                SELECT * FROM failure_records
                WHERE project_id IS ? AND cluster_key = ? AND status IN ('observed', 'linked', 'verified')
                ORDER BY last_seen_at DESC LIMIT 1
                """,
                (project_id, cluster_key),
            ).fetchone()
            if row is None:
                failure_id = new_id()
                self.connection.execute(
                    """
                    INSERT INTO failure_records(
                      id, project_id, run_id, task_id, signature, cluster_key, raw_error,
                      metadata_json, first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        failure_id,
                        project_id,
                        run_id,
                        task_id,
                        signature,
                        cluster_key,
                        raw_error,
                        json.dumps(metadata or {}, sort_keys=True),
                        timestamp,
                        timestamp,
                    ),
                )
            else:
                failure_id = str(row["id"])
                merged_metadata = json.loads(row["metadata_json"] or "{}")
                merged_metadata.update(metadata or {})
                self.connection.execute(
                    """
                    UPDATE failure_records
                    SET occurrence_count = occurrence_count + 1, last_seen_at = ?,
                        raw_error = ?, run_id = COALESCE(?, run_id),
                        task_id = COALESCE(?, task_id), metadata_json = ?
                    WHERE id = ?
                    """,
                    (
                        timestamp,
                        raw_error,
                        run_id,
                        task_id,
                        json.dumps(merged_metadata, sort_keys=True),
                        failure_id,
                    ),
                )
            self.connection.commit()
        result = self.get_failure(failure_id)
        if result is None:
            raise RuntimeError("failure record did not round-trip")
        return result

    def link_failure_resolution(
        self,
        failure_id: str,
        *,
        root_cause: str,
        patch_id: str,
        verification_ref: str,
        verified: bool,
    ) -> dict[str, Any]:
        status = "verified" if verified else "linked"
        self.connection.execute(
            """
            UPDATE failure_records
            SET root_cause = ?, patch_id = ?, verification_ref = ?, status = ?, last_seen_at = ?
            WHERE id = ?
            """,
            (root_cause, patch_id, verification_ref, status, now_unix(), failure_id),
        )
        self.connection.commit()
        result = self.get_failure(failure_id)
        if result is None:
            raise KeyError(f"unknown failure record: {failure_id}")
        return result

    def get_failure(self, failure_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM failure_records WHERE id = ?", (failure_id,)
        ).fetchone()
        data = row_to_dict(row)
        if data is None:
            return None
        data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        return data

    def list_failure_clusters(self, project_id: str | None = None) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT cluster_key, signature, SUM(occurrence_count) AS occurrence_count,
                   MAX(last_seen_at) AS last_seen_at,
                   SUM(CASE WHEN status = 'verified' THEN occurrence_count ELSE 0 END) AS verified_count
            FROM failure_records
            WHERE project_id IS ?
            GROUP BY cluster_key, signature
            ORDER BY occurrence_count DESC, last_seen_at DESC
            """,
            (project_id,),
        ).fetchall()
        return [row_to_dict(row) or {} for row in rows]

    def insert_learning_candidate(
        self,
        *,
        project_id: str | None,
        run_id: str | None,
        patch_id: str | None,
        kind: str,
        prompt: str,
        chosen: str,
        verification: dict[str, Any],
        score: float,
    ) -> dict[str, Any]:
        candidate_id = new_id()
        self.connection.execute(
            """
            INSERT INTO learning_candidates(
              id, project_id, run_id, patch_id, kind, status, prompt, chosen,
              verification_json, score, created_at
            ) VALUES (?, ?, ?, ?, ?, 'pending_review', ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                project_id,
                run_id,
                patch_id,
                kind,
                prompt,
                chosen,
                json.dumps(verification, sort_keys=True),
                score,
                now_unix(),
            ),
        )
        self.connection.commit()
        row = self.connection.execute(
            "SELECT * FROM learning_candidates WHERE id = ?", (candidate_id,)
        ).fetchone()
        result = row_to_dict(row)
        if result is None:
            raise RuntimeError("learning candidate insert did not round-trip")
        result["verification"] = json.loads(result.pop("verification_json") or "{}")
        return result

    def attach_failure_candidate(self, failure_id: str, candidate_id: str) -> None:
        self.connection.execute(
            "UPDATE failure_records SET dataset_candidate_id = ? WHERE id = ? AND status = 'verified'",
            (candidate_id, failure_id),
        )
        self.connection.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return row_to_dict(row)

    def create_patch_record(
        self,
        *,
        project_id: str,
        run_id: str | None,
        status: str,
        diff: str,
        files_changed: list[str],
        backup: dict[str, Any],
    ) -> dict[str, Any]:
        if self.get_project(project_id) is None:
            raise KeyError(f"unknown project: {project_id}")
        patch_id = new_id()
        timestamp = now_unix()
        self.connection.execute(
            """
            INSERT INTO patches(id, project_id, run_id, status, diff, files_changed_json, backup_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patch_id,
                project_id,
                run_id,
                status,
                diff,
                json.dumps(files_changed, sort_keys=True),
                json.dumps(backup, sort_keys=True),
                timestamp,
            ),
        )
        self.connection.commit()
        patch = self.get_patch(patch_id)
        if patch is None:
            raise RuntimeError("patch insert did not round-trip")
        return patch

    def get_patch(self, patch_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM patches WHERE id = ?", (patch_id,)).fetchone()
        data = row_to_dict(row)
        if data is None:
            return None
        data["files_changed"] = json.loads(data.pop("files_changed_json") or "[]")
        data["backup"] = json.loads(data.pop("backup_json") or "{}")
        return data

    def update_patch_status(
        self, patch_id: str, status: str, *, applied: bool = False, rolled_back: bool = False
    ) -> None:
        timestamp = now_unix()
        self.connection.execute(
            """
            UPDATE patches
            SET status = ?,
                applied_at = CASE WHEN ? THEN ? ELSE applied_at END,
                rolled_back_at = CASE WHEN ? THEN ? ELSE rolled_back_at END
            WHERE id = ?
            """,
            (status, applied, timestamp, rolled_back, timestamp, patch_id),
        )
        self.connection.commit()

    def clear_index(self, project_id: str) -> None:
        self.connection.execute("DELETE FROM code_chunks WHERE project_id = ?", (project_id,))
        self.connection.execute("DELETE FROM workspace_files WHERE project_id = ?", (project_id,))
        self.connection.commit()

    def list_workspace_file_hashes(self, project_id: str) -> dict[str, str]:
        return {
            str(row["path"]): str(row["content_hash"])
            for row in self.connection.execute(
                "SELECT path, content_hash FROM workspace_files WHERE project_id = ?",
                (project_id,),
            ).fetchall()
        }

    def begin_index_revision(
        self,
        *,
        project_id: str,
        source_revision: str,
        source_snapshot_sha256: str,
        chunker_version: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        project = self.get_project(project_id)
        if project is None:
            raise KeyError(f"unknown project: {project_id}")
        if len(source_snapshot_sha256) != 64:
            raise ValueError("source_snapshot_sha256 must be a SHA-256 hex digest")
        revision_id = new_id()
        timestamp = now_unix()
        with self._lock:
            try:
                self.connection.execute(
                    """
                    INSERT INTO rag_index_revisions(
                      id, organization_id, project_id, source_revision,
                      source_snapshot_sha256, chunker_version, status,
                      manifest_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'building', ?, ?)
                    """,
                    (
                        revision_id,
                        project["organization_id"],
                        project_id,
                        source_revision,
                        source_snapshot_sha256,
                        chunker_version,
                        json.dumps(manifest, sort_keys=True),
                        timestamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise RuntimeError(
                    "another index generation is already building for this project"
                ) from exc
            self.connection.execute(
                "UPDATE projects SET index_status = 'indexing', index_error = NULL, updated_at = ? WHERE id = ?",
                (timestamp, project_id),
            )
            self.connection.commit()
        revision = self.get_index_revision(revision_id)
        if revision is None:
            raise RuntimeError("index revision insert did not round-trip")
        return revision

    def get_index_revision(self, revision_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM rag_index_revisions WHERE id = ?",
            (revision_id,),
        ).fetchone()
        result = row_to_dict(row)
        if result is not None:
            result["manifest"] = json.loads(result.pop("manifest_json") or "{}")
        return result

    def active_index_revision(self, project_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT r.* FROM rag_index_revisions r
            JOIN projects p ON p.active_index_revision = r.id
            WHERE p.id = ? AND r.status = 'committed'
            """,
            (project_id,),
        ).fetchone()
        result = row_to_dict(row)
        if result is not None:
            result["manifest"] = json.loads(result.pop("manifest_json") or "{}")
        return result

    def fail_index_revision(self, revision_id: str, error: str) -> None:
        safe_error = error[:4096]
        with self._lock:
            row = self.connection.execute(
                "SELECT project_id, status FROM rag_index_revisions WHERE id = ?",
                (revision_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown index revision: {revision_id}")
            if row["status"] != "building":
                raise RuntimeError("only a building index revision can fail")
            timestamp = now_unix()
            self.connection.execute(
                "UPDATE rag_index_revisions SET status = 'failed', error = ? WHERE id = ?",
                (safe_error, revision_id),
            )
            self.connection.execute(
                "UPDATE projects SET index_status = 'failed', index_error = ?, updated_at = ? WHERE id = ?",
                (safe_error, timestamp, row["project_id"]),
            )
            self.connection.commit()

    def commit_index_revision(
        self,
        *,
        revision_id: str,
        files: Iterable[dict[str, Any]],
        chunks: Iterable[dict[str, Any]],
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        file_rows = list(files)
        chunk_rows = list(chunks)
        file_ids = {str(item["id"]) for item in file_rows}
        if len(file_ids) != len(file_rows):
            raise ValueError("index revision contains duplicate file IDs")
        chunk_ids = {str(item["id"]) for item in chunk_rows}
        if len(chunk_ids) != len(chunk_rows):
            raise ValueError("index revision contains duplicate chunk IDs")
        if any(str(item["file_id"]) not in file_ids for item in chunk_rows):
            raise ValueError("index revision contains a chunk with an unknown file ID")

        with self._lock:
            self.connection.commit()
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                revision = self.connection.execute(
                    "SELECT * FROM rag_index_revisions WHERE id = ?",
                    (revision_id,),
                ).fetchone()
                if revision is None:
                    raise KeyError(f"unknown index revision: {revision_id}")
                if revision["status"] != "building":
                    raise RuntimeError("only a building index revision can be committed")
                project_id = str(revision["project_id"])
                timestamp = now_unix()
                self.connection.execute(
                    "DELETE FROM code_chunks WHERE project_id = ?", (project_id,)
                )
                self.connection.execute(
                    "DELETE FROM workspace_files WHERE project_id = ?", (project_id,)
                )
                self.connection.executemany(
                    """
                    INSERT INTO workspace_files(
                      id, project_id, path, language, content_hash, size_bytes,
                      index_revision, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item["id"],
                            project_id,
                            item["path"],
                            item.get("language"),
                            item["content_hash"],
                            item["size_bytes"],
                            revision_id,
                            timestamp,
                        )
                        for item in file_rows
                    ],
                )
                self.connection.executemany(
                    """
                    INSERT INTO code_chunks(
                      id, project_id, file_id, path, language, start_line, end_line,
                      symbol_name, kind, chunk_hash, token_count, content,
                      metadata_json, index_revision, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item["id"],
                            project_id,
                            item["file_id"],
                            item["path"],
                            item.get("language"),
                            item["start_line"],
                            item["end_line"],
                            item.get("symbol_name"),
                            item["kind"],
                            item["chunk_hash"],
                            item["token_count"],
                            item["content"],
                            json.dumps(item.get("metadata", {}), sort_keys=True),
                            revision_id,
                            timestamp,
                        )
                        for item in chunk_rows
                    ],
                )
                self.connection.execute(
                    """
                    UPDATE rag_index_revisions SET status = 'superseded'
                    WHERE project_id = ? AND status = 'committed' AND id != ?
                    """,
                    (project_id, revision_id),
                )
                self.connection.execute(
                    """
                    UPDATE rag_index_revisions
                    SET status = 'committed', manifest_json = ?, committed_at = ?, error = NULL
                    WHERE id = ?
                    """,
                    (json.dumps(manifest, sort_keys=True), timestamp, revision_id),
                )
                self.connection.execute(
                    """
                    UPDATE projects
                    SET active_index_revision = ?, index_status = 'ready', index_error = NULL,
                        last_indexed_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (revision_id, timestamp, timestamp, project_id),
                )
                outbox_id = new_id()
                self.connection.execute(
                    """
                    INSERT INTO rag_outbox_events(
                      id, organization_id, project_id, revision_id, kind,
                      payload_json, status, available_at, created_at
                    ) VALUES (?, ?, ?, ?, 'vector_sync_required', ?, 'pending', ?, ?)
                    """,
                    (
                        outbox_id,
                        revision["organization_id"],
                        project_id,
                        revision_id,
                        json.dumps(
                            {"revision_id": revision_id, "manifest": manifest}, sort_keys=True
                        ),
                        timestamp,
                        timestamp,
                    ),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        committed = self.get_index_revision(revision_id)
        if committed is None:
            raise RuntimeError("committed index revision did not round-trip")
        return committed

    @staticmethod
    def _rag_job_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        result = row_to_dict(row)
        if result is None:
            return None
        result["request"] = json.loads(result.pop("request_json") or "{}")
        result["result"] = json.loads(result.pop("result_json") or "{}")
        result["cancel_requested"] = bool(result.get("cancel_requested"))
        return result

    @staticmethod
    def _rag_outbox_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        result = row_to_dict(row)
        if result is None:
            return None
        result["payload"] = json.loads(result.pop("payload_json") or "{}")
        return result

    def create_index_job(
        self,
        *,
        organization_id: str,
        project_id: str,
        idempotency_key: str,
        request: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        self.require_project_access(project_id, organization_id)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", idempotency_key):
            raise ValueError("idempotency_key must contain 8-128 safe characters")
        if max_attempts < 1 or max_attempts > 20:
            raise ValueError("max_attempts must be between 1 and 20")
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 64 * 1024:
            raise ValueError("index job request exceeds 64 KiB")
        timestamp = now_unix()
        job_id = new_id()
        with self._lock:
            existing = self.connection.execute(
                "SELECT * FROM rag_index_jobs WHERE organization_id = ? AND idempotency_key = ?",
                (organization_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                job = self._rag_job_row(existing)
                if (
                    job is None
                    or json.dumps(job["request"], sort_keys=True, separators=(",", ":")) != encoded
                ):
                    raise ValueError("idempotency key is already bound to a different request")
                return job
            self.connection.execute(
                """
                INSERT INTO rag_index_jobs(
                  id, organization_id, project_id, idempotency_key, status,
                  max_attempts, request_json, result_json, available_at,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'queued', ?, ?, '{}', ?, ?, ?)
                """,
                (
                    job_id,
                    organization_id,
                    project_id,
                    idempotency_key,
                    max_attempts,
                    encoded,
                    timestamp,
                    timestamp,
                    timestamp,
                ),
            )
            self.connection.commit()
        job = self.get_index_job(job_id, organization_id=organization_id)
        if job is None:
            raise RuntimeError("index job insert did not round-trip")
        return job

    def get_index_job(
        self, job_id: str, *, organization_id: str | None = None
    ) -> dict[str, Any] | None:
        sql = "SELECT * FROM rag_index_jobs WHERE id = ?"
        parameters: list[Any] = [job_id]
        if organization_id is not None:
            sql += " AND organization_id = ?"
            parameters.append(organization_id)
        return self._rag_job_row(self.connection.execute(sql, parameters).fetchone())

    def list_index_jobs(
        self,
        *,
        organization_id: str,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = min(max(limit, 1), 500)
        sql = "SELECT * FROM rag_index_jobs WHERE organization_id = ?"
        parameters: list[Any] = [organization_id]
        if project_id is not None:
            sql += " AND project_id = ?"
            parameters.append(project_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(limit)
        return [
            self._rag_job_row(row) for row in self.connection.execute(sql, parameters).fetchall()
        ]  # type: ignore[misc]

    def claim_index_job(self, *, worker_id: str, lease_seconds: int = 60) -> dict[str, Any] | None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,127}", worker_id):
            raise ValueError("worker_id contains unsafe characters")
        lease_seconds = min(max(lease_seconds, 10), 900)
        timestamp = now_unix()
        with self._lock:
            self.connection.commit()
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                row = self.connection.execute(
                    """
                    SELECT * FROM rag_index_jobs
                    WHERE (
                      (status = 'queued' AND available_at <= ?)
                      OR (status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?)
                    )
                    AND cancel_requested = 0 AND attempt < max_attempts
                    ORDER BY available_at, created_at
                    LIMIT 1
                    """,
                    (timestamp, timestamp),
                ).fetchone()
                if row is None:
                    self.connection.commit()
                    return None
                self.connection.execute(
                    """
                    UPDATE rag_index_jobs
                    SET status = 'running', attempt = attempt + 1, lease_owner = ?,
                        lease_expires_at = ?, updated_at = ?, error = NULL
                    WHERE id = ?
                    """,
                    (worker_id, timestamp + lease_seconds, timestamp, row["id"]),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        return self.get_index_job(str(row["id"]))

    def heartbeat_index_job(self, job_id: str, *, worker_id: str, lease_seconds: int = 60) -> bool:
        timestamp = now_unix()
        cursor = self.connection.execute(
            """
            UPDATE rag_index_jobs SET lease_expires_at = ?, updated_at = ?
            WHERE id = ? AND status = 'running' AND lease_owner = ? AND cancel_requested = 0
            """,
            (timestamp + min(max(lease_seconds, 10), 900), timestamp, job_id, worker_id),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def index_job_cancel_requested(self, job_id: str) -> bool:
        row = self.connection.execute(
            "SELECT cancel_requested, status FROM rag_index_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        return row is None or bool(row["cancel_requested"]) or row["status"] == "cancelled"

    def request_index_job_cancel(self, job_id: str, *, organization_id: str) -> dict[str, Any]:
        timestamp = now_unix()
        cursor = self.connection.execute(
            """
            UPDATE rag_index_jobs
            SET cancel_requested = 1,
                status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END,
                finished_at = CASE WHEN status = 'queued' THEN ? ELSE finished_at END,
                updated_at = ?
            WHERE id = ? AND organization_id = ? AND status IN ('queued', 'running')
            """,
            (timestamp, timestamp, job_id, organization_id),
        )
        self.connection.commit()
        job = self.get_index_job(job_id, organization_id=organization_id)
        if job is None:
            raise KeyError("index job not found")
        if cursor.rowcount == 0 and job["status"] not in {
            "cancelled",
            "succeeded",
            "failed",
            "dead_letter",
        }:
            raise RuntimeError("index job cannot be cancelled from its current state")
        return job

    def complete_index_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        revision_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
        timestamp = now_unix()
        cursor = self.connection.execute(
            """
            UPDATE rag_index_jobs
            SET status = 'succeeded', revision_id = ?, result_json = ?, error = NULL,
                lease_owner = NULL, lease_expires_at = NULL, updated_at = ?, finished_at = ?
            WHERE id = ? AND status = 'running' AND lease_owner = ? AND cancel_requested = 0
            """,
            (revision_id, encoded, timestamp, timestamp, job_id, worker_id),
        )
        self.connection.commit()
        if cursor.rowcount != 1:
            raise RuntimeError("index job lease was lost or cancellation was requested")
        job = self.get_index_job(job_id)
        assert job is not None
        return job

    def fail_index_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        error: str,
        retry_delay_seconds: float,
        permanent: bool = False,
    ) -> dict[str, Any]:
        timestamp = now_unix()
        with self._lock:
            self.connection.commit()
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                row = self.connection.execute(
                    "SELECT * FROM rag_index_jobs WHERE id = ? AND status = 'running' AND lease_owner = ?",
                    (job_id, worker_id),
                ).fetchone()
                if row is None:
                    raise RuntimeError("index job lease was lost")
                cancelled = bool(row["cancel_requested"])
                exhausted = int(row["attempt"]) >= int(row["max_attempts"])
                status = (
                    "cancelled"
                    if cancelled
                    else "dead_letter"
                    if permanent or exhausted
                    else "queued"
                )
                finished_at = timestamp if status in {"cancelled", "dead_letter"} else None
                self.connection.execute(
                    """
                    UPDATE rag_index_jobs
                    SET status = ?, error = ?, lease_owner = NULL, lease_expires_at = NULL,
                        available_at = ?, updated_at = ?, finished_at = ?
                    WHERE id = ?
                    """,
                    (
                        status,
                        error[:4096],
                        timestamp + max(0.0, retry_delay_seconds),
                        timestamp,
                        finished_at,
                        job_id,
                    ),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        job = self.get_index_job(job_id)
        assert job is not None
        return job

    def claim_rag_outbox(self, *, worker_id: str, lease_seconds: int = 60) -> dict[str, Any] | None:
        timestamp = now_unix()
        with self._lock:
            self.connection.commit()
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                row = self.connection.execute(
                    """
                    SELECT * FROM rag_outbox_events
                    WHERE (
                      (status = 'pending' AND available_at <= ?)
                      OR (status = 'delivering' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?)
                    ) AND attempt < max_attempts
                    ORDER BY available_at, created_at LIMIT 1
                    """,
                    (timestamp, timestamp),
                ).fetchone()
                if row is None:
                    self.connection.commit()
                    return None
                self.connection.execute(
                    """
                    UPDATE rag_outbox_events
                    SET status = 'delivering', attempt = attempt + 1, lease_owner = ?,
                        lease_expires_at = ?, error = NULL
                    WHERE id = ?
                    """,
                    (worker_id, timestamp + min(max(lease_seconds, 10), 900), row["id"]),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        return self._rag_outbox_row(
            self.connection.execute(
                "SELECT * FROM rag_outbox_events WHERE id = ?", (row["id"],)
            ).fetchone()
        )

    def complete_rag_outbox(self, event_id: str, *, worker_id: str) -> None:
        timestamp = now_unix()
        cursor = self.connection.execute(
            """
            UPDATE rag_outbox_events SET status = 'delivered', delivered_at = ?,
              lease_owner = NULL, lease_expires_at = NULL, error = NULL
            WHERE id = ? AND status = 'delivering' AND lease_owner = ?
            """,
            (timestamp, event_id, worker_id),
        )
        self.connection.commit()
        if cursor.rowcount != 1:
            raise RuntimeError("RAG outbox lease was lost")

    def fail_rag_outbox(
        self, event_id: str, *, worker_id: str, error: str, retry_delay_seconds: float
    ) -> None:
        timestamp = now_unix()
        with self._lock:
            row = self.connection.execute(
                "SELECT attempt, max_attempts FROM rag_outbox_events WHERE id = ? AND lease_owner = ?",
                (event_id, worker_id),
            ).fetchone()
            if row is None:
                raise RuntimeError("RAG outbox lease was lost")
            status = "dead_letter" if int(row["attempt"]) >= int(row["max_attempts"]) else "pending"
            self.connection.execute(
                """
                UPDATE rag_outbox_events SET status = ?, error = ?, available_at = ?,
                  lease_owner = NULL, lease_expires_at = NULL
                WHERE id = ?
                """,
                (status, error[:4096], timestamp + max(0.0, retry_delay_seconds), event_id),
            )
            self.connection.commit()

    def upsert_workspace_file(
        self,
        *,
        project_id: str,
        path: str,
        language: str | None,
        content_hash: str,
        size_bytes: int,
    ) -> str:
        file_id = new_id()
        timestamp = now_unix()
        self.connection.execute(
            """
            INSERT INTO workspace_files(id, project_id, path, language, content_hash, size_bytes, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, path) DO UPDATE SET
              language = excluded.language,
              content_hash = excluded.content_hash,
              size_bytes = excluded.size_bytes,
              indexed_at = excluded.indexed_at
            """,
            (file_id, project_id, path, language, content_hash, size_bytes, timestamp),
        )
        row = self.connection.execute(
            "SELECT id FROM workspace_files WHERE project_id = ? AND path = ?",
            (project_id, path),
        ).fetchone()
        self.connection.commit()
        if row is None:
            raise RuntimeError(f"workspace file upsert failed for {path}")
        return str(row["id"])

    def insert_code_chunks(self, chunks: Iterable[dict[str, Any]]) -> int:
        count = 0
        for chunk in chunks:
            self.connection.execute(
                """
                INSERT INTO code_chunks(
                  id, project_id, file_id, path, language, start_line, end_line,
                  symbol_name, kind, chunk_hash, token_count, content, metadata_json, indexed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk["id"],
                    chunk["project_id"],
                    chunk["file_id"],
                    chunk["path"],
                    chunk.get("language"),
                    chunk["start_line"],
                    chunk["end_line"],
                    chunk.get("symbol_name"),
                    chunk["kind"],
                    chunk["chunk_hash"],
                    chunk["token_count"],
                    chunk["content"],
                    json.dumps(chunk.get("metadata", {}), sort_keys=True),
                    now_unix(),
                ),
            )
            count += 1
        self.connection.commit()
        return count

    def list_chunks(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT c.*, f.content_hash AS file_hash
            FROM code_chunks c
            JOIN workspace_files f ON f.id = c.file_id
            WHERE c.project_id = ?
            ORDER BY c.path, c.start_line
            """,
            (project_id,),
        ).fetchall()
        return [self._chunk_row(row) for row in rows]

    def get_chunk(self, chunk_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        sql = """
            SELECT c.*, f.content_hash AS file_hash
            FROM code_chunks c JOIN workspace_files f ON f.id = c.file_id
            WHERE c.id = ?
        """
        params: list[Any] = [chunk_id]
        if project_id is not None:
            sql += " AND c.project_id = ?"
            params.append(project_id)
        row = self.connection.execute(sql, params).fetchone()
        return self._chunk_row(row) if row is not None else None

    def index_status(self, project_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        if project is None:
            raise KeyError(f"unknown project: {project_id}")
        files = self.connection.execute(
            "SELECT COUNT(*) AS count FROM workspace_files WHERE project_id = ?",
            (project_id,),
        ).fetchone()["count"]
        chunks = self.connection.execute(
            "SELECT COUNT(*) AS count FROM code_chunks WHERE project_id = ?",
            (project_id,),
        ).fetchone()["count"]
        symbols = self.connection.execute(
            "SELECT COUNT(*) AS count FROM code_chunks WHERE project_id = ? AND symbol_name IS NOT NULL",
            (project_id,),
        ).fetchone()["count"]
        return {
            "project_id": project_id,
            "status": project["index_status"],
            "active_index_revision": project.get("active_index_revision"),
            "index_error": project.get("index_error"),
            "file_count": int(files),
            "chunk_count": int(chunks),
            "symbol_count": int(symbols),
            "last_indexed_at": project["last_indexed_at"],
        }

    def _chunk_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = row_to_dict(row) or {}
        metadata = data.pop("metadata_json", "{}")
        data["metadata"] = json.loads(metadata or "{}")
        return data

    def insert_memory_entry(
        self,
        *,
        project_id: str | None,
        kind: str,
        content: dict[str, Any],
        source_run_id: str | None = None,
        relevance: float = 0.5,
        success_rate: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry_id = new_id()
        timestamp = now_unix()
        organization_id = "local"
        if project_id is not None:
            project = self.get_project(project_id)
            if project is None:
                raise KeyError(f"unknown project: {project_id}")
            organization_id = str(project["organization_id"])
        self.connection.execute(
            """
            INSERT INTO memory_entries(
              id, organization_id, project_id, kind, content, source_run_id, relevance,
              success_rate, usage_count, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                entry_id,
                organization_id,
                project_id,
                kind,
                json.dumps(content, sort_keys=True),
                source_run_id,
                relevance,
                success_rate,
                json.dumps(metadata or {}, sort_keys=True),
                timestamp,
            ),
        )
        self.connection.commit()
        result = self.get_memory_entry(entry_id)
        if result is None:
            raise RuntimeError("memory entry insert did not round-trip")
        return result

    def get_memory_entry(self, entry_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM memory_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        data = row_to_dict(row)
        if data is None:
            return None
        data["content"] = json.loads(data["content"] or "{}")
        data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        return data

    def list_memory_entries(
        self, project_id: str | None = None, *, kinds: list[str] | None = None
    ) -> list[dict[str, Any]]:
        params: list[Any] = []
        if project_id is not None:
            project = self.get_project(project_id)
            if project is None:
                raise KeyError(f"unknown project: {project_id}")
            query = """
                SELECT * FROM memory_entries
                WHERE organization_id = ? AND (project_id = ? OR project_id IS NULL)
                ORDER BY created_at DESC
            """
            params.extend([project["organization_id"], project_id])
        else:
            query = "SELECT * FROM memory_entries WHERE organization_id = 'local' ORDER BY created_at DESC"
        rows = self.connection.execute(query, params).fetchall()
        allowed_kinds = set(kinds or [])
        results: list[dict[str, Any]] = []
        for row in rows:
            data = row_to_dict(row) or {}
            if allowed_kinds and data.get("kind") not in allowed_kinds:
                continue
            data["content"] = json.loads(data["content"] or "{}")
            data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
            results.append(data)
        return results

    def mark_memory_used(self, entry_id: str) -> None:
        self.connection.execute(
            """
            UPDATE memory_entries
            SET usage_count = usage_count + 1, last_used_at = ?
            WHERE id = ?
            """,
            (now_unix(), entry_id),
        )
        self.connection.commit()


class PostgresRAGStore:
    """Synchronous request-scoped Postgres authority for Hybrid RAG.

    Every connection transaction binds the immutable organization UUID with
    ``set_config`` before touching an RLS-protected table. Instances must not be
    shared across organizations.
    """

    def __init__(
        self,
        dsn: str,
        *,
        organization_id: str,
        min_pool_size: int = 1,
        max_pool_size: int = 20,
    ) -> None:
        if not dsn.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError("PostgresRAGStore requires a postgresql:// DSN")
        try:
            self.organization_id = str(uuid.UUID(organization_id))
        except ValueError as exc:
            raise ValueError("production organization_id must be a UUID") from exc
        if min_pool_size < 1 or max_pool_size < min_pool_size or max_pool_size > 100:
            raise ValueError("invalid Postgres RAG pool bounds")
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - readiness handles it.
            raise RuntimeError(
                "psycopg and psycopg_pool are required for Postgres RAG persistence"
            ) from exc
        normalized_dsn = dsn.replace("postgresql+psycopg://", "postgresql://", 1)
        self._pool = ConnectionPool(
            conninfo=normalized_dsn,
            min_size=min_pool_size,
            max_size=max_pool_size,
            kwargs={"row_factory": dict_row, "connect_timeout": 5},
            timeout=10,
            open=True,
        )
        self._owns_pool = True

    @classmethod
    def from_shared_pool(cls, pool: Any, *, organization_id: str) -> PostgresRAGStore:
        """Create an organization-bound facade over one process-wide pool."""

        instance = cls.__new__(cls)
        try:
            instance.organization_id = str(uuid.UUID(organization_id))
        except ValueError as exc:
            raise ValueError("production organization_id must be a UUID") from exc
        if pool is None or not callable(getattr(pool, "connection", None)):
            raise TypeError("shared Postgres pool must provide connection()")
        instance._pool = pool
        instance._owns_pool = False
        return instance

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        with self._pool.connection() as connection, connection.transaction():
            connection.execute(
                "SELECT set_config('craftly.organization_id', %s, true)",
                (self.organization_id,),
            )
            yield connection

    @staticmethod
    def _json(value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)
        return value

    @classmethod
    def _project_row(cls, row: Any | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    @classmethod
    def _revision_row(cls, row: Any | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result["manifest"] = cls._json(result.get("manifest") or {})
        return result

    @classmethod
    def _chunk_row(cls, row: Any | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result["id"] = str(result["id"])
        result["project_id"] = str(result["project_id"])
        result["file_id"] = str(result["file_id"])
        result["index_revision"] = (
            str(result["index_revision"]) if result.get("index_revision") else None
        )
        result["metadata"] = cls._json(result.pop("metadata", {}) or {})
        return result

    @classmethod
    def _job_row(cls, row: Any | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for key in ("id", "organization_id", "project_id", "revision_id"):
            if result.get(key) is not None:
                result[key] = str(result[key])
        result["request"] = cls._json(result.pop("request_json", {}) or {})
        result["result"] = cls._json(result.pop("result_json", {}) or {})
        return result

    @classmethod
    def _outbox_row(cls, row: Any | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for key in ("id", "organization_id", "project_id", "revision_id"):
            result[key] = str(result[key])
        result["payload"] = cls._json(result.pop("payload", {}) or {})
        return result

    def ensure_organization(
        self, organization_id: str, *, name: str | None = None
    ) -> dict[str, Any]:
        if str(uuid.UUID(organization_id)) != self.organization_id:
            raise PermissionError("organization binding mismatch")
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO organizations(id, name, status) VALUES (%s::uuid, %s, 'active')
                ON CONFLICT(id) DO UPDATE SET name = COALESCE(NULLIF(EXCLUDED.name, ''), organizations.name)
                RETURNING *
                """,
                (self.organization_id, name or self.organization_id),
            ).fetchone()
        assert row is not None
        return dict(row)

    def add_organization_member(
        self, organization_id: str, user_id: str, *, role: str = "member"
    ) -> None:
        if organization_id != self.organization_id:
            raise PermissionError("organization binding mismatch")
        if role not in {"owner", "admin", "member", "viewer"}:
            raise ValueError("unsupported organization role")
        self.ensure_organization(organization_id)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO organization_members(organization_id,user_id,role)
                VALUES (%s::uuid,%s,%s)
                ON CONFLICT(organization_id,user_id) DO UPDATE SET role=EXCLUDED.role
                """,
                (organization_id, user_id, role),
            )

    def create_project(
        self,
        *,
        name: str,
        repo_path: str,
        default_branch: str = "main",
        organization_id: str | None = None,
        owner_user_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        organization_id = organization_id or self.organization_id
        if organization_id != self.organization_id:
            raise PermissionError("organization binding mismatch")
        self.ensure_organization(organization_id)
        identifier = str(uuid.UUID(project_id)) if project_id else str(uuid.uuid4())
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO projects(id,organization_id,name,repo_path,default_branch,index_status)
                VALUES (%s::uuid,%s::uuid,%s,%s,%s,'not_indexed') RETURNING *
                """,
                (identifier, organization_id, name, str(Path(repo_path).resolve()), default_branch),
            ).fetchone()
            if owner_user_id:
                connection.execute(
                    """
                    INSERT INTO organization_members(organization_id,user_id,role)
                    VALUES (%s::uuid,%s,'owner') ON CONFLICT(organization_id,user_id) DO NOTHING
                    """,
                    (organization_id, owner_user_id),
                )
                connection.execute(
                    """
                    INSERT INTO project_members(project_id,organization_id,user_id,role)
                    VALUES (%s::uuid,%s::uuid,%s,'owner')
                    """,
                    (identifier, organization_id, owner_user_id),
                )
        assert row is not None
        return dict(row)

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE id=%s::uuid AND organization_id=%s::uuid",
                (project_id, self.organization_id),
            ).fetchone()
        return self._project_row(row)

    def require_project_access(
        self,
        project_id: str,
        organization_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        if organization_id != self.organization_id:
            raise PermissionError("organization binding mismatch")
        with self._connection() as connection:
            if user_id is None:
                row = connection.execute(
                    "SELECT * FROM projects WHERE id=%s::uuid AND organization_id=%s::uuid",
                    (project_id, organization_id),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT p.* FROM projects p JOIN project_members m ON m.project_id=p.id
                    WHERE p.id=%s::uuid AND p.organization_id=%s::uuid
                      AND m.organization_id=%s::uuid AND m.user_id=%s
                    """,
                    (project_id, organization_id, organization_id, user_id),
                ).fetchone()
        if row is None:
            raise PermissionError("project is unavailable to this organization member")
        return dict(row)

    def list_workspace_file_hashes(self, project_id: str) -> dict[str, str]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT path,content_hash FROM workspace_files WHERE project_id=%s::uuid",
                (project_id,),
            ).fetchall()
        return {str(row["path"]): str(row["content_hash"]) for row in rows}

    def begin_index_revision(self, **kwargs: Any) -> dict[str, Any]:
        project = self.get_project(str(kwargs["project_id"]))
        if project is None:
            raise KeyError("unknown project")
        revision_id = str(uuid.uuid4())
        try:
            from psycopg.types.json import Jsonb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("psycopg is required") from exc
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO rag_index_revisions(
                  id,organization_id,project_id,source_revision,source_snapshot_sha256,
                  chunker_version,status,manifest
                ) VALUES (%s::uuid,%s::uuid,%s::uuid,%s,%s,%s,'building',%s) RETURNING *
                """,
                (
                    revision_id,
                    self.organization_id,
                    kwargs["project_id"],
                    kwargs["source_revision"],
                    kwargs["source_snapshot_sha256"],
                    kwargs["chunker_version"],
                    Jsonb(kwargs["manifest"]),
                ),
            ).fetchone()
            connection.execute(
                "UPDATE projects SET index_status='indexing',index_error=NULL,updated_at=now() WHERE id=%s::uuid",
                (kwargs["project_id"],),
            )
        result = self._revision_row(row)
        assert result is not None
        return result

    def get_index_revision(self, revision_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM rag_index_revisions WHERE id=%s::uuid",
                (revision_id,),
            ).fetchone()
        return self._revision_row(row)

    def active_index_revision(self, project_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT r.* FROM rag_index_revisions r JOIN projects p ON p.active_index_revision=r.id
                WHERE p.id=%s::uuid AND r.status='committed'
                """,
                (project_id,),
            ).fetchone()
        return self._revision_row(row)

    def fail_index_revision(self, revision_id: str, error: str) -> None:
        with self._connection() as connection:
            row = connection.execute(
                """
                UPDATE rag_index_revisions SET status='failed',error=%s
                WHERE id=%s::uuid AND status='building' RETURNING project_id
                """,
                (error[:4096], revision_id),
            ).fetchone()
            if row is None:
                raise RuntimeError("only a building revision can fail")
            connection.execute(
                "UPDATE projects SET index_status='failed',index_error=%s,updated_at=now() WHERE id=%s::uuid",
                (error[:4096], row["project_id"]),
            )

    def commit_index_revision(
        self,
        *,
        revision_id: str,
        files: Iterable[dict[str, Any]],
        chunks: Iterable[dict[str, Any]],
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        file_rows = list(files)
        chunk_rows = list(chunks)
        file_ids = {str(item["id"]) for item in file_rows}
        if len(file_ids) != len(file_rows) or len({str(item["id"]) for item in chunk_rows}) != len(
            chunk_rows
        ):
            raise ValueError("index revision contains duplicate IDs")
        if any(str(item["file_id"]) not in file_ids for item in chunk_rows):
            raise ValueError("index revision chunk references an unknown file")
        try:
            from psycopg.types.json import Jsonb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("psycopg is required") from exc
        with self._connection() as connection:
            revision = connection.execute(
                "SELECT * FROM rag_index_revisions WHERE id=%s::uuid FOR UPDATE",
                (revision_id,),
            ).fetchone()
            if revision is None or revision["status"] != "building":
                raise RuntimeError("only a building revision can be committed")
            project_id = str(revision["project_id"])
            connection.execute("DELETE FROM code_chunks WHERE project_id=%s::uuid", (project_id,))
            connection.execute(
                "DELETE FROM workspace_files WHERE project_id=%s::uuid", (project_id,)
            )
            with connection.cursor().copy(
                """
                COPY workspace_files(id,project_id,path,language,content_hash,size_bytes,index_revision,indexed_at)
                FROM STDIN
                """
            ) as copy:
                for item in file_rows:
                    copy.write_row(
                        (
                            item["id"],
                            project_id,
                            item["path"],
                            item.get("language"),
                            item["content_hash"],
                            item["size_bytes"],
                            revision_id,
                            datetime.now(UTC),
                        )
                    )
            with connection.cursor().copy(
                """
                COPY code_chunks(
                  id,project_id,file_id,path,language,start_line,end_line,symbol_name,kind,
                  chunk_hash,token_count,content,metadata,index_revision,indexed_at
                ) FROM STDIN
                """
            ) as copy:
                for item in chunk_rows:
                    copy.write_row(
                        (
                            item["id"],
                            project_id,
                            item["file_id"],
                            item["path"],
                            item.get("language"),
                            item["start_line"],
                            item["end_line"],
                            item.get("symbol_name"),
                            item["kind"],
                            item["chunk_hash"],
                            item["token_count"],
                            item["content"],
                            Jsonb(item.get("metadata", {})),
                            revision_id,
                            datetime.now(UTC),
                        )
                    )
            connection.execute(
                "UPDATE rag_index_revisions SET status='superseded' WHERE project_id=%s::uuid AND status='committed'",
                (project_id,),
            )
            row = connection.execute(
                """
                UPDATE rag_index_revisions SET status='committed',manifest=%s,error=NULL,committed_at=now()
                WHERE id=%s::uuid RETURNING *
                """,
                (Jsonb(manifest), revision_id),
            ).fetchone()
            connection.execute(
                """
                UPDATE projects SET active_index_revision=%s::uuid,index_status='ready',index_error=NULL,
                  last_indexed_at=now(),updated_at=now() WHERE id=%s::uuid
                """,
                (revision_id, project_id),
            )
            connection.execute(
                """
                INSERT INTO rag_outbox_events(id,organization_id,project_id,revision_id,kind,payload)
                VALUES (%s::uuid,%s::uuid,%s::uuid,%s::uuid,'vector_sync_required',%s)
                """,
                (
                    str(uuid.uuid4()),
                    self.organization_id,
                    project_id,
                    revision_id,
                    Jsonb({"revision_id": revision_id, "manifest": manifest}),
                ),
            )
        result = self._revision_row(row)
        assert result is not None
        return result

    def list_chunks(self, project_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT c.*,f.content_hash AS file_hash FROM code_chunks c
                JOIN workspace_files f ON f.id=c.file_id WHERE c.project_id=%s::uuid
                ORDER BY c.path,c.start_line
                """,
                (project_id,),
            ).fetchall()
        return [self._chunk_row(row) for row in rows]  # type: ignore[misc]

    def get_chunk(self, chunk_id: str, *, project_id: str | None = None) -> dict[str, Any] | None:
        sql = """
          SELECT c.*,f.content_hash AS file_hash FROM code_chunks c
          JOIN workspace_files f ON f.id=c.file_id WHERE c.id=%s::uuid
        """
        parameters: list[Any] = [chunk_id]
        if project_id is not None:
            sql += " AND c.project_id=%s::uuid"
            parameters.append(project_id)
        with self._connection() as connection:
            row = connection.execute(sql, parameters).fetchone()
        return self._chunk_row(row)

    def index_status(self, project_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        if project is None:
            raise KeyError("unknown project")
        with self._connection() as connection:
            counts = connection.execute(
                """
                SELECT (SELECT count(*) FROM workspace_files WHERE project_id=%s::uuid) AS file_count,
                       (SELECT count(*) FROM code_chunks WHERE project_id=%s::uuid) AS chunk_count
                """,
                (project_id, project_id),
            ).fetchone()
            latest = connection.execute(
                "SELECT status,error FROM rag_index_jobs WHERE project_id=%s::uuid ORDER BY created_at DESC LIMIT 1",
                (project_id,),
            ).fetchone()
        return {
            "project_id": project_id,
            "status": project["index_status"],
            "active_index_revision": str(project["active_index_revision"])
            if project.get("active_index_revision")
            else None,
            "index_error": project.get("index_error"),
            "file_count": int(counts["file_count"]),
            "chunk_count": int(counts["chunk_count"]),
            "latest_job": dict(latest) if latest else None,
        }

    def create_index_job(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs["organization_id"] != self.organization_id:
            raise PermissionError("organization binding mismatch")
        self.require_project_access(kwargs["project_id"], self.organization_id)
        key = str(kwargs["idempotency_key"])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", key):
            raise ValueError("invalid idempotency key")
        try:
            from psycopg.types.json import Jsonb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("psycopg is required") from exc
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO rag_index_jobs(
                  id,organization_id,project_id,idempotency_key,status,max_attempts,request_json,result_json
                ) VALUES (%s::uuid,%s::uuid,%s::uuid,%s,'queued',%s,%s,'{}')
                ON CONFLICT(organization_id,idempotency_key) DO NOTHING RETURNING *
                """,
                (
                    str(uuid.uuid4()),
                    self.organization_id,
                    kwargs["project_id"],
                    key,
                    kwargs.get("max_attempts", 3),
                    Jsonb(kwargs.get("request") or {}),
                ),
            ).fetchone()
            if row is None:
                row = connection.execute(
                    "SELECT * FROM rag_index_jobs WHERE organization_id=%s::uuid AND idempotency_key=%s",
                    (self.organization_id, key),
                ).fetchone()
                existing = self._job_row(row)
                if existing is None or existing["request"] != (kwargs.get("request") or {}):
                    raise ValueError("idempotency key is bound to a different request")
                return existing
        result = self._job_row(row)
        assert result is not None
        return result

    def get_index_job(
        self, job_id: str, *, organization_id: str | None = None
    ) -> dict[str, Any] | None:
        if organization_id is not None and organization_id != self.organization_id:
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM rag_index_jobs WHERE id=%s::uuid", (job_id,)
            ).fetchone()
        return self._job_row(row)

    def list_index_jobs(
        self, *, organization_id: str, project_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        if organization_id != self.organization_id:
            raise PermissionError("organization binding mismatch")
        sql = "SELECT * FROM rag_index_jobs WHERE organization_id=%s::uuid"
        parameters: list[Any] = [organization_id]
        if project_id:
            sql += " AND project_id=%s::uuid"
            parameters.append(project_id)
        sql += " ORDER BY created_at DESC LIMIT %s"
        parameters.append(min(max(limit, 1), 500))
        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._job_row(row) for row in rows]  # type: ignore[misc]

    def claim_index_job(self, *, worker_id: str, lease_seconds: int = 60) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                WITH candidate AS (
                  SELECT id FROM rag_index_jobs
                  WHERE ((status='queued' AND available_at<=now()) OR
                         (status='running' AND lease_expires_at<=now()))
                    AND cancel_requested=false AND attempt<max_attempts
                  ORDER BY available_at,created_at FOR UPDATE SKIP LOCKED LIMIT 1
                )
                UPDATE rag_index_jobs j SET status='running',attempt=j.attempt+1,lease_owner=%s,
                  lease_expires_at=now()+(%s * interval '1 second'),updated_at=now(),error=NULL
                FROM candidate WHERE j.id=candidate.id RETURNING j.*
                """,
                (worker_id, min(max(lease_seconds, 10), 900)),
            ).fetchone()
        return self._job_row(row)

    def heartbeat_index_job(self, job_id: str, *, worker_id: str, lease_seconds: int = 60) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                """
                UPDATE rag_index_jobs SET lease_expires_at=now()+(%s*interval '1 second'),updated_at=now()
                WHERE id=%s::uuid AND status='running' AND lease_owner=%s AND cancel_requested=false RETURNING id
                """,
                (min(max(lease_seconds, 10), 900), job_id, worker_id),
            ).fetchone()
        return row is not None

    def index_job_cancel_requested(self, job_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT cancel_requested,status FROM rag_index_jobs WHERE id=%s::uuid",
                (job_id,),
            ).fetchone()
        return row is None or bool(row["cancel_requested"]) or row["status"] == "cancelled"

    def request_index_job_cancel(self, job_id: str, *, organization_id: str) -> dict[str, Any]:
        if organization_id != self.organization_id:
            raise PermissionError("organization binding mismatch")
        with self._connection() as connection:
            row = connection.execute(
                """
                UPDATE rag_index_jobs SET cancel_requested=true,
                  status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,
                  finished_at=CASE WHEN status='queued' THEN now() ELSE finished_at END,updated_at=now()
                WHERE id=%s::uuid AND status IN ('queued','running') RETURNING *
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            current = self.get_index_job(job_id)
            if current is None:
                raise KeyError("index job not found")
            return current
        result = self._job_row(row)
        assert result is not None
        return result

    def complete_index_job(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
        try:
            from psycopg.types.json import Jsonb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("psycopg is required") from exc
        with self._connection() as connection:
            row = connection.execute(
                """
                UPDATE rag_index_jobs SET status='succeeded',revision_id=%s::uuid,result_json=%s,
                  error=NULL,lease_owner=NULL,lease_expires_at=NULL,updated_at=now(),finished_at=now()
                WHERE id=%s::uuid AND status='running' AND lease_owner=%s AND cancel_requested=false RETURNING *
                """,
                (
                    kwargs["revision_id"],
                    Jsonb(kwargs.get("result") or {}),
                    job_id,
                    kwargs["worker_id"],
                ),
            ).fetchone()
        if row is None:
            raise RuntimeError("index job lease was lost or cancellation was requested")
        result = self._job_row(row)
        assert result is not None
        return result

    def fail_index_job(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
        with self._connection() as connection:
            current = connection.execute(
                "SELECT * FROM rag_index_jobs WHERE id=%s::uuid AND status='running' AND lease_owner=%s FOR UPDATE",
                (job_id, kwargs["worker_id"]),
            ).fetchone()
            if current is None:
                raise RuntimeError("index job lease was lost")
            cancelled = bool(current["cancel_requested"])
            exhausted = int(current["attempt"]) >= int(current["max_attempts"])
            status = (
                "cancelled"
                if cancelled
                else "dead_letter"
                if kwargs.get("permanent") or exhausted
                else "queued"
            )
            row = connection.execute(
                """
                UPDATE rag_index_jobs SET status=%s,error=%s,lease_owner=NULL,lease_expires_at=NULL,
                  available_at=now()+(%s*interval '1 second'),updated_at=now(),
                  finished_at=CASE WHEN %s IN ('cancelled','dead_letter') THEN now() ELSE NULL END
                WHERE id=%s::uuid RETURNING *
                """,
                (
                    status,
                    str(kwargs["error"])[:4096],
                    max(0.0, kwargs["retry_delay_seconds"]),
                    status,
                    job_id,
                ),
            ).fetchone()
        result = self._job_row(row)
        assert result is not None
        return result

    def claim_rag_outbox(self, *, worker_id: str, lease_seconds: int = 60) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                WITH candidate AS (
                  SELECT id FROM rag_outbox_events
                  WHERE ((status='pending' AND available_at<=now()) OR
                         (status='delivering' AND lease_expires_at<=now()))
                    AND attempt<max_attempts
                  ORDER BY available_at,created_at FOR UPDATE SKIP LOCKED LIMIT 1
                )
                UPDATE rag_outbox_events e SET status='delivering',attempt=e.attempt+1,lease_owner=%s,
                  lease_expires_at=now()+(%s*interval '1 second'),error=NULL
                FROM candidate WHERE e.id=candidate.id RETURNING e.*
                """,
                (worker_id, min(max(lease_seconds, 10), 900)),
            ).fetchone()
        return self._outbox_row(row)

    def complete_rag_outbox(self, event_id: str, *, worker_id: str) -> None:
        with self._connection() as connection:
            row = connection.execute(
                """
                UPDATE rag_outbox_events SET status='delivered',delivered_at=now(),lease_owner=NULL,
                  lease_expires_at=NULL,error=NULL WHERE id=%s::uuid AND status='delivering' AND lease_owner=%s RETURNING id
                """,
                (event_id, worker_id),
            ).fetchone()
        if row is None:
            raise RuntimeError("RAG outbox lease was lost")

    def fail_rag_outbox(self, event_id: str, **kwargs: Any) -> None:
        with self._connection() as connection:
            current = connection.execute(
                "SELECT * FROM rag_outbox_events WHERE id=%s::uuid AND lease_owner=%s FOR UPDATE",
                (event_id, kwargs["worker_id"]),
            ).fetchone()
            if current is None:
                raise RuntimeError("RAG outbox lease was lost")
            status = (
                "dead_letter"
                if int(current["attempt"]) >= int(current["max_attempts"])
                else "pending"
            )
            connection.execute(
                """
                UPDATE rag_outbox_events SET status=%s,error=%s,available_at=now()+(%s*interval '1 second'),
                  lease_owner=NULL,lease_expires_at=NULL WHERE id=%s::uuid
                """,
                (
                    status,
                    str(kwargs["error"])[:4096],
                    max(0.0, kwargs["retry_delay_seconds"]),
                    event_id,
                ),
            )

    def insert_memory_entry(self, **kwargs: Any) -> dict[str, Any]:
        try:
            from psycopg.types.json import Jsonb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("psycopg is required") from exc
        project_id = kwargs.get("project_id")
        if project_id:
            self.require_project_access(project_id, self.organization_id)
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO memory_entries(
                  id,organization_id,project_id,kind,content,source_run_id,relevance,success_rate,metadata
                ) VALUES (%s::uuid,%s::uuid,%s::uuid,%s,%s,%s::uuid,%s,%s,%s) RETURNING *
                """,
                (
                    str(uuid.uuid4()),
                    self.organization_id,
                    project_id,
                    kwargs["kind"],
                    json.dumps(kwargs["content"], sort_keys=True),
                    kwargs.get("source_run_id"),
                    kwargs.get("relevance", 0.5),
                    kwargs.get("success_rate", 0.5),
                    Jsonb(kwargs.get("metadata") or {}),
                ),
            ).fetchone()
        result = dict(row)
        result["id"] = str(result["id"])
        result["content"] = self._json(result["content"])
        result["metadata"] = self._json(result["metadata"])
        return result

    def list_memory_entries(
        self, project_id: str | None = None, *, kinds: list[str] | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM memory_entries WHERE organization_id=%s::uuid"
        parameters: list[Any] = [self.organization_id]
        if project_id:
            sql += " AND (project_id=%s::uuid OR project_id IS NULL)"
            parameters.append(project_id)
        if kinds:
            sql += " AND kind=ANY(%s::text[])"
            parameters.append(kinds)
        sql += " ORDER BY created_at DESC"
        with self._connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["id"] = str(item["id"])
            item["project_id"] = str(item["project_id"]) if item.get("project_id") else None
            item["content"] = self._json(item["content"])
            item["metadata"] = self._json(item["metadata"])
            results.append(item)
        return results

    def mark_memory_used(self, entry_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE memory_entries SET usage_count=usage_count+1,last_used_at=now() WHERE id=%s::uuid",
                (entry_id,),
            )

    def close(self) -> None:
        if self._owns_pool:
            self._pool.close()

    def __enter__(self) -> PostgresRAGStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class PostgresRAGStoreFactory:
    """One bounded connection pool serving immutable tenant-scoped facades."""

    def __init__(self, dsn: str, *, min_pool_size: int = 1, max_pool_size: int = 20) -> None:
        if not dsn.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError("PostgresRAGStoreFactory requires a postgresql:// DSN")
        if min_pool_size < 1 or max_pool_size < min_pool_size or max_pool_size > 100:
            raise ValueError("invalid shared Postgres RAG pool bounds")
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - readiness handles it.
            raise RuntimeError(
                "psycopg and psycopg_pool are required for Postgres RAG persistence"
            ) from exc
        self.dsn = dsn.replace("postgresql+psycopg://", "postgresql://", 1)
        self._pool = ConnectionPool(
            conninfo=self.dsn,
            min_size=min_pool_size,
            max_size=max_pool_size,
            kwargs={"row_factory": dict_row, "connect_timeout": 5},
            timeout=10,
            open=True,
        )

    def for_organization(self, organization_id: str) -> PostgresRAGStore:
        return PostgresRAGStore.from_shared_pool(self._pool, organization_id=organization_id)

    def check(self) -> None:
        self._pool.check()

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> PostgresRAGStoreFactory:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class PostgresRAGDispatcher:
    """Least-surface global queue claimer for multi-tenant RAG workers.

    The dispatcher can only execute the two migration-owned claim functions.
    All repository reads and writes continue through an organization-bound
    ``PostgresRAGStore`` with RLS enabled.
    """

    def __init__(self, dsn: str, *, max_pool_size: int = 4) -> None:
        if not dsn.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError("PostgresRAGDispatcher requires a postgresql:// DSN")
        if max_pool_size < 1 or max_pool_size > 16:
            raise ValueError("dispatcher pool size must be between 1 and 16")
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - readiness handles it.
            raise RuntimeError("psycopg and psycopg_pool are required for RAG dispatch") from exc
        self._pool = ConnectionPool(
            conninfo=dsn.replace("postgresql+psycopg://", "postgresql://", 1),
            min_size=1,
            max_size=max_pool_size,
            kwargs={"row_factory": dict_row, "connect_timeout": 5},
            timeout=10,
            open=True,
        )

    @staticmethod
    def _validate_worker(worker_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{2,127}", worker_id):
            raise ValueError("worker_id contains unsafe characters")

    def claim_index_job(self, *, worker_id: str, lease_seconds: int = 120) -> dict[str, Any] | None:
        self._validate_worker(worker_id)
        lease_seconds = min(max(lease_seconds, 10), 900)
        with self._pool.connection() as connection, connection.transaction():
            row = connection.execute(
                "SELECT * FROM craftly_claim_rag_index_job(%s, %s)",
                (worker_id, lease_seconds),
            ).fetchone()
        return PostgresRAGStore._job_row(row)

    def claim_outbox(self, *, worker_id: str, lease_seconds: int = 120) -> dict[str, Any] | None:
        self._validate_worker(worker_id)
        lease_seconds = min(max(lease_seconds, 10), 900)
        with self._pool.connection() as connection, connection.transaction():
            row = connection.execute(
                "SELECT * FROM craftly_claim_rag_outbox(%s, %s)",
                (worker_id, lease_seconds),
            ).fetchone()
        return PostgresRAGStore._outbox_row(row)

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> PostgresRAGDispatcher:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
