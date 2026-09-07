from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.craftly.db import LocalStore
from src.craftly.gateway import api as gateway_api
from src.craftly.indexing import ContextBuilder, RepositoryIndexer
from src.craftly.indexing.context_builder import query_terms
from src.craftly.runtime.taskgraph import AgentRunCreateRequest, TaskCompleteRequest, TaskGraphRuntime
from src.craftly.verifier import VerificationRequest, VerifierRuntime


class CraftlyMvpFoundationTest(unittest.TestCase):
    def test_repository_index_and_context_builder_find_relevant_auth_code(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            workspace = Path(workspace_dir)
            (workspace / "src").mkdir()
            (workspace / "tests").mkdir()
            (workspace / "src" / "auth.py").write_text(
                "\n".join(
                    [
                        "import hashlib",
                        "",
                        "def hash_password(password: str) -> str:",
                        "    return hashlib.sha256(password.encode()).hexdigest()",
                        "",
                        "def login_user(username: str, password: str) -> bool:",
                        "    return username == 'admin' and bool(hash_password(password))",
                    ]
                ),
                encoding="utf-8",
            )
            (workspace / "tests" / "test_auth.py").write_text(
                "from src.auth import login_user\n\n"
                "def test_login_user():\n"
                "    assert login_user('admin', 'secret')\n",
                encoding="utf-8",
            )

            with LocalStore(Path(db_dir) / "craftly.sqlite3") as store:
                project = store.create_project(name="demo", repo_path=str(workspace))

                index_report = RepositoryIndexer(store).index_project(project_id=project["id"])
                self.assertEqual(index_report.status, "ready")
                self.assertGreaterEqual(index_report.file_count, 2)
                self.assertGreaterEqual(index_report.symbol_count, 2)

                context = ContextBuilder(store).build(
                    project_id=project["id"],
                    query="fix login password hashing bug in auth",
                    token_budget=4000,
                )
                self.assertGreater(context.estimated_tokens, 1)
                self.assertTrue(any(chunk.path == "src/auth.py" for chunk in context.chunks))
                self.assertIn("login_user", context.prompt_context)
                auth_chunks = store.list_chunks(project["id"])
                login_chunk = next(chunk for chunk in auth_chunks if chunk.get("symbol_name") == "login_user")
                self.assertIn("hash_password", login_chunk["metadata"]["calls"])
                self.assertIn("hashlib", login_chunk["metadata"]["imports"])
                scored_login = ContextBuilder(store).score_chunk(login_chunk, query_terms("hash_password"), set())
                self.assertIn("dependency", scored_login["reason"])

                hash_chunk = next(chunk for chunk in auth_chunks if chunk.get("symbol_name") == "hash_password")
                self.assertEqual(hash_chunk["metadata"]["signature"], "def hash_password(password: str) -> str")

                run = TaskGraphRuntime(store).create_agent_run(
                    AgentRunCreateRequest(
                        project_id=project["id"],
                        prompt="fix login password hashing bug",
                        mode="debug",
                    )
                )
                self.assertEqual(run.status, "queued")
                graph = store.get_task_graph(run.task_graph_id)
                self.assertIsNotNone(graph)
                self.assertEqual(len(graph["nodes"]), 10)
                self.assertIn("planner", [node["kind"] for node in graph["nodes"]])
                self.assertIn("critic_review", [node["kind"] for node in graph["nodes"]])
                self.assertIn("security_review", [node["kind"] for node in graph["nodes"]])
                self.assertIn("performance_review", [node["kind"] for node in graph["nodes"]])
                self.assertEqual(graph["nodes"][0]["kind"], "understand")
                self.assertEqual(graph["nodes"][-1]["kind"], "summarize")
                runtime = TaskGraphRuntime(store)
                advance = runtime.advance(run.task_graph_id)
                self.assertEqual(advance.status, "running")
                self.assertEqual(advance.active_task["kind"], "understand")
                after_complete = runtime.complete_task(
                    advance.active_task["id"],
                    TaskCompleteRequest(outputs={"intent": "debug_auth"}),
                )
                self.assertEqual(after_complete.status, "running")
                self.assertEqual(after_complete.completed_task_count, 1)
                self.assertEqual(after_complete.active_task["kind"], "planner")

                verification = VerifierRuntime(store).run(
                    VerificationRequest(
                        project_id=project["id"],
                        commands=[["python", "-c", "print('ok')"]],
                        run_secret_scan=True,
                        run_semgrep=True,
                    )
                )
                self.assertEqual(verification.verdict, "accept")
                self.assertTrue(any(item["tool"] == "semgrep" for item in verification.security_results))

    def test_gateway_project_index_and_context_contract(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as db_dir:
            workspace = Path(workspace_dir)
            (workspace / "auth.py").write_text(
                "def login_user(username, password):\n"
                "    return username == 'admin' and password == 'secret'\n",
                encoding="utf-8",
            )
            original_store = gateway_api.STORE
            replacement_store = LocalStore(Path(db_dir) / "gateway.sqlite3")
            gateway_api.STORE = replacement_store
            try:
                client = TestClient(gateway_api.app)
                with patch.dict("os.environ", {"CRAFTLY_PROJECT_ROOTS": str(workspace)}, clear=False):
                    create_response = client.post(
                        "/v1/projects",
                        json={"name": "gateway-demo", "repo_path": ".", "default_branch": "main"},
                    )
                self.assertEqual(create_response.status_code, 200, create_response.text)
                project_id = create_response.json()["id"]

                index_response = client.post(f"/v1/projects/{project_id}/index", json={"force": True})
                self.assertEqual(index_response.status_code, 200, index_response.text)
                self.assertEqual(index_response.json()["status"], "ready")

                symbols_response = client.get(f"/v1/projects/{project_id}/symbols")
                self.assertEqual(symbols_response.status_code, 200, symbols_response.text)
                self.assertEqual(symbols_response.json()["symbol_count"], 1)
                self.assertEqual(symbols_response.json()["symbols"][0]["symbol_name"], "login_user")

                context_response = client.post(
                    "/v1/context/build",
                    json={
                        "project_id": project_id,
                        "query": "login auth password",
                        "token_budget": 4000,
                        "max_chunks": 5,
                    },
                )
                self.assertEqual(context_response.status_code, 200, context_response.text)
                payload = context_response.json()
                self.assertIn("login_user", payload["prompt_context"])
                self.assertTrue(payload["chunks"])

                session_response = client.post(
                    "/v1/sessions",
                    json={"project_id": project_id, "title": "debug auth"},
                )
                self.assertEqual(session_response.status_code, 200, session_response.text)
                session_id = session_response.json()["id"]

                run_response = client.post(
                    "/v1/agent/runs",
                    json={
                        "project_id": project_id,
                        "session_id": session_id,
                        "prompt": "fix login auth password bug",
                        "mode": "debug",
                    },
                )
                self.assertEqual(run_response.status_code, 200, run_response.text)
                task_graph_id = run_response.json()["task_graph_id"]

                graph_response = client.get(f"/v1/taskgraphs/{task_graph_id}")
                self.assertEqual(graph_response.status_code, 200, graph_response.text)
                self.assertEqual(len(graph_response.json()["nodes"]), 10)

                advance_response = client.post(f"/v1/taskgraphs/{task_graph_id}/advance")
                self.assertEqual(advance_response.status_code, 200, advance_response.text)
                active_task_id = advance_response.json()["active_task"]["id"]
                self.assertEqual(advance_response.json()["active_task"]["kind"], "understand")

                complete_response = client.post(
                    f"/v1/tasks/{active_task_id}/complete",
                    json={"outputs": {"intent": "code_edit"}},
                )
                self.assertEqual(complete_response.status_code, 200, complete_response.text)
                self.assertEqual(complete_response.json()["completed_task_count"], 1)

                failing = client.post(
                    "/v1/tools/execute",
                    json={
                        "project_id": project_id,
                        "run_id": run_response.json()["run_id"],
                        "tool": "test",
                        "command": ["python", "-c", "import auth; raise SystemExit(0 if auth.login_user('admin', 'secret') else 1)"],
                    },
                )
                self.assertEqual(failing.status_code, 200, failing.text)
                self.assertEqual(failing.json()["status"], "ok")

                patch_response = client.post(
                    "/v1/patches/preview",
                    json={
                        "project_id": project_id,
                        "run_id": run_response.json()["run_id"],
                        "edits": [
                            {
                                "path": "auth.py",
                                "new_content": (
                                    "def login_user(username, password):\n"
                                    "    return username == 'admin' and bool(password)\n"
                                ),
                            }
                        ],
                    },
                )
                self.assertEqual(patch_response.status_code, 200, patch_response.text)
                patch_id = patch_response.json()["patch_id"]
                self.assertIn("-    return username == 'admin' and password == 'secret'", patch_response.json()["diff"])

                apply_response = client.post(f"/v1/patches/{patch_id}/apply")
                self.assertEqual(apply_response.status_code, 200, apply_response.text)
                self.assertEqual(apply_response.json()["status"], "applied")

                verifier_response = client.post(
                    "/v1/verifier/run",
                    json={
                        "project_id": project_id,
                        "run_id": run_response.json()["run_id"],
                        "patch_id": patch_id,
                        "commands": [["python", "-c", "import auth; raise SystemExit(0 if auth.login_user('admin', 'changed') else 1)"]],
                        "run_secret_scan": True,
                    },
                )
                self.assertEqual(verifier_response.status_code, 200, verifier_response.text)
                self.assertEqual(verifier_response.json()["verdict"], "accept")

                rollback_response = client.post(f"/v1/patches/{patch_id}/rollback")
                self.assertEqual(rollback_response.status_code, 200, rollback_response.text)
                self.assertEqual(rollback_response.json()["status"], "rolled_back")
            finally:
                replacement_store.close()
                gateway_api.STORE = original_store


if __name__ == "__main__":
    unittest.main()

