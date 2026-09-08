"""Unit and integration tests for Craftly Agentic Tool-Use and Sandbox Verification."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.craftly.learning.agentic_dataset import (
    AgenticScenario,
    ToolCall,
    ToolResult,
    build_canonical_agentic_scenarios,
    format_tool_call,
    format_tool_result,
    parse_agentic_output,
)
from src.craftly.model_ops.backends import ModelBackend
from src.craftly.runtime.agent_loop import (
    AgenticRuntime,
    AgenticTask,
    apply_unified_diff,
)


class MockAgenticModelBackend(ModelBackend):
    """Deterministic mock backend for agentic flow testing."""

    name: str = "mock_agentic"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return "I have completed analysis."


class TestCraftlyAgentic(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_agentic_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_tool_call_formatting_and_parsing(self) -> None:
        call = ToolCall(tool="test", command=["pytest", "tests/test_vuln.py"], timeout_ms=10000)
        formatted = format_tool_call(call)
        self.assertIn("<|tool_call|>", formatted)
        self.assertIn("<|tool_end|>", formatted)

        parsed = parse_agentic_output(
            f"<|thought_start|>I should run tests first.<|thought_end|>\n{formatted}"
        )
        self.assertEqual(parsed.thought, "I should run tests first.")
        self.assertIsNotNone(parsed.tool_call)
        self.assertEqual(parsed.tool_call.tool, "test")
        self.assertEqual(parsed.tool_call.command, ["pytest", "tests/test_vuln.py"])
        self.assertFalse(parsed.is_terminal)

    def test_tool_result_formatting(self) -> None:
        res = ToolResult(tool="test", status="ok", exit_code=0, stdout="1 passed")
        formatted = format_tool_result(res)
        self.assertIn("<|tool_result|>", formatted)
        self.assertIn("1 passed", formatted)

    def test_apply_unified_diff_clean(self) -> None:
        sample_file = self.test_dir / "calc.py"
        sample_file.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

        diff = (
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a - b\n"
            "+    return a + b\n"
        )
        applied = apply_unified_diff(self.test_dir, diff)
        self.assertTrue(applied)
        updated_content = sample_file.read_text(encoding="utf-8")
        self.assertIn("return a + b", updated_content)

    def test_canonical_agentic_scenarios(self) -> None:
        scenarios = build_canonical_agentic_scenarios()
        self.assertGreaterEqual(len(scenarios), 3)
        cwe_ids = {s.cwe_id for s in scenarios}
        self.assertIn("CWE-89", cwe_ids)
        self.assertIn("CWE-22", cwe_ids)
        self.assertIn("CWE-208", cwe_ids)

    async def _async_test_agent_runtime_loop(self) -> None:
        # Create a workspace with a failing test and fixable code
        ws = self.test_dir / "workspace"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "vuln.py").write_text("def is_secure():\n    return False\n", encoding="utf-8")
        (ws / "test_vuln.py").write_text(
            "import unittest\nfrom vuln import is_secure\n\n"
            "class TestVuln(unittest.TestCase):\n"
            "    def test_flag(self):\n"
            "        self.assertTrue(is_secure())\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )

        # Mock trajectory:
        # Turn 1: Call tool to run test (fails)
        # Turn 2: Synthesize patch (passes)
        responses = [
            "<|thought_start|>Running test.<|thought_end|>\n"
            '<|tool_call|>{"tool": "test", "command": ["python", "test_vuln.py"]}<|tool_end|>',
            "<|thought_start|>Test failed. Applying patch.<|thought_end|>\n"
            "<|patch_start|>\n"
            "--- a/vuln.py\n"
            "+++ b/vuln.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def is_secure():\n"
            "-    return False\n"
            "+    return True\n"
            "<|patch_end|>",
        ]

        backend = MockAgenticModelBackend(responses)
        runtime = AgenticRuntime(backend=backend)

        task = AgenticTask(
            task_id="test-patch-and-verify",
            workspace_dir=str(ws),
            instruction="Fix is_secure to return True and verify test passes.",
            verification_command=["python", "test_vuln.py"],
            max_turns=3,
        )

        report = await runtime.run_task(task)
        self.assertEqual(report.status, "verified_pass")
        self.assertEqual(report.turns_taken, 2)
        self.assertIsNotNone(report.patch)
        self.assertIn("return True", report.patch)
        self.assertEqual(report.verification_exit_code, 0)

    def test_agent_runtime_loop(self) -> None:
        import asyncio
        asyncio.run(self._async_test_agent_runtime_loop())


if __name__ == "__main__":
    unittest.main()
