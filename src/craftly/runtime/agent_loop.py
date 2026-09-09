"""Autonomous agentic execution runtime with isolated sandbox verification.

Coordinates model reasoning, allowlisted tool execution, unified diff patch
application, and test verification in an ephemeral scratch workspace.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess  # nosec B404 - execution bounded by strict argv and timeout policy
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.learning.agentic_dataset import (
    PATCH_END,
    PATCH_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    ToolCall,
    ToolResult,
    format_tool_result,
    parse_agentic_output,
)
from src.craftly.model_ops.backends import ModelBackend
from src.craftly.shared.schemas import StrictModel
from src.craftly.tools.policy import ALLOWED_TOOL_COMMANDS


class AgenticTask(StrictModel):
    """A coding or security task assigned to the agent."""

    task_id: str
    workspace_dir: str
    instruction: str
    verification_command: list[str] = Field(min_length=1)
    max_turns: int = Field(default=4, ge=1, le=10)


class AgenticStep(StrictModel):
    """Record of a single turn in the agent trajectory."""

    turn: int
    thought: str | None = None
    tool_call: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    patch: str | None = None
    explanation: str | None = None


class AgenticExecutionReport(StrictModel):
    """Final outcome and provenance of an agent run."""

    task_id: str
    status: str = Field(pattern=r"^(verified_pass|regression_detected|max_turns_exceeded|patch_application_failed|error)$")
    turns_taken: int
    patch: str | None = None
    verification_stdout: str = ""
    verification_stderr: str = ""
    verification_exit_code: int | None = None
    steps: list[AgenticStep] = Field(default_factory=list)
    duration_ms: float = 0.0


def apply_patch_hunks(original_lines: list[str], hunks: list[tuple[int, list[str], list[str]]]) -> list[str]:
    """Apply unified diff hunks (old_start, old_lines, new_lines) deterministically."""
    result = list(original_lines)
    for old_start, old_block, new_block in sorted(hunks, key=lambda h: h[0], reverse=True):
        # 1-indexed to 0-indexed
        idx = max(0, old_start - 1)
        # Verify match or find closest match in neighborhood
        match_idx = None
        for offset in range(0, max(len(result), 10)):
            for candidate in [idx + offset, idx - offset]:
                if 0 <= candidate <= len(result) - len(old_block):
                    if result[candidate : candidate + len(old_block)] == old_block:
                        match_idx = candidate
                        break
            if match_idx is not None:
                break

        if match_idx is not None:
            result[match_idx : match_idx + len(old_block)] = new_block
        else:
            # Fallback simple string replacement if exact hunk header shifted
            joined_old = "".join(old_block)
            joined_res = "".join(result)
            if joined_old in joined_res:
                replaced = joined_res.replace(joined_old, "".join(new_block), 1)
                result = [line + "\n" for line in replaced.splitlines()]
    return result


def apply_unified_diff(workspace_dir: Path, diff_text: str) -> bool:
    """Parse unified diff and apply cleanly across affected files."""
    file_diffs = re.split(r"(?=^--- )", diff_text, flags=re.MULTILINE)
    applied_any = False

    for chunk in file_diffs:
        chunk = chunk.strip()
        if not chunk.startswith("--- "):
            continue

        match = re.search(r"^--- [a-zA-Z0-9_./\\]+[\t ]*(?:[a-zA-Z0-9_./\\-]+)?\r?\n\+\+\+ ([a-zA-Z0-9_./\\]+)", chunk, re.MULTILINE)
        if not match:
            continue

        raw_rel = match.group(1)
        if raw_rel.startswith("b/") or raw_rel.startswith("b\\"):
            raw_rel = raw_rel[2:]

        rel_path = raw_rel.replace("\\", "/").lstrip("/")
        target_file = (workspace_dir / rel_path).resolve()

        if not target_file.is_relative_to(workspace_dir.resolve()):
            continue

        # Parse hunks from chunk
        hunks: list[tuple[int, list[str], list[str]]] = []
        hunk_chunks = re.split(r"(?=^@@ )", chunk, flags=re.MULTILINE)

        for h in hunk_chunks:
            if not h.startswith("@@ "):
                continue
            h_header = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", h)
            if not h_header:
                continue
            old_start = int(h_header.group(1))
            old_lines: list[str] = []
            new_lines: list[str] = []

            for line in h.splitlines()[1:]:
                if line.startswith("-"):
                    old_lines.append(line[1:] + "\n")
                elif line.startswith("+"):
                    new_lines.append(line[1:] + "\n")
                elif line.startswith(" "):
                    old_lines.append(line[1:] + "\n")
                    new_lines.append(line[1:] + "\n")

            hunks.append((old_start, old_lines, new_lines))

        if hunks and target_file.exists():
            original_content = target_file.read_text(encoding="utf-8").splitlines(keepends=True)
            modified_content = apply_patch_hunks(original_content, hunks)
            target_file.write_text("".join(modified_content), encoding="utf-8")
            applied_any = True

    return applied_any


def apply_patch_to_workspace(workspace_dir: Path, patch_text: str) -> bool:
    """Parse patch text and apply to workspace files.

    Supports:
    1. Standard Unified Diff (--- a/file ... +++ b/file ... @@ ... @@)
    2. File-annotated code blocks (e.g. ### File: db.py or # File: db.py)
    3. Function-level replacement matching existing functions in workspace .py files
    4. Whole-file replacement if a single non-test source file exists
    """
    cleaned = patch_text.strip()
    if not cleaned:
        return False

    # Strip markdown code blocks if wrapped
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # 1. Try standard unified diff
    if "--- " in cleaned and "+++ " in cleaned:
        if apply_unified_diff(workspace_dir, cleaned):
            return True

    # 2. File header pattern: e.g. "### File: `db.py`" or "# File: db.py"
    file_match = re.search(r"(?:###?\s*File:?\s*[`'\"]?|#\s*file:?\s*[`'\"]?)([a-zA-Z0-9_./\\]+\.py)[`'\"]?", cleaned, re.IGNORECASE)
    if file_match:
        target_name = file_match.group(1).replace("\\", "/").split("/")[-1]
        target_file = workspace_dir / target_name
        if target_file.exists():
            code_lines = [l for l in cleaned.splitlines() if not re.search(r"(?:###?\s*File|#\s*file)", l, re.IGNORECASE)]
            target_file.write_text("\n".join(code_lines).strip() + "\n", encoding="utf-8")
            return True

    # 3. Match function definition inside workspace Python files
    func_names = re.findall(r"def\s+([a-zA-Z0-9_]+)\s*\(", cleaned)
    if func_names:
        for p in sorted(workspace_dir.glob("*.py")):
            if p.name.startswith("test_"):
                continue
            original_code = p.read_text(encoding="utf-8", errors="replace")
            if any(f"def {fn}(" in original_code for fn in func_names):
                new_code = original_code
                replaced_any = False
                for fn in func_names:
                    fn_pattern = rf"(def\s+{fn}\s*\([^)]*\).*?)(?=\n(?:def|class)\s+|\Z)"
                    patch_fn_m = re.search(fn_pattern, cleaned, re.DOTALL)
                    orig_fn_m = re.search(fn_pattern, new_code, re.DOTALL)
                    if patch_fn_m and orig_fn_m:
                        new_code = new_code.replace(orig_fn_m.group(0), patch_fn_m.group(0), 1)
                        replaced_any = True

                if replaced_any:
                    for line in cleaned.splitlines():
                        line_s = line.strip()
                        if (line_s.startswith("import ") or line_s.startswith("from ")) and line_s not in new_code:
                            new_code = line_s + "\n" + new_code
                    p.write_text(new_code, encoding="utf-8")
                    return True

    # 4. If exactly one non-test .py file in workspace and patch contains python code
    source_files = [p for p in workspace_dir.glob("*.py") if not p.name.startswith("test_")]
    if len(source_files) == 1 and ("def " in cleaned or "import " in cleaned or "from " in cleaned):
        source_files[0].write_text(cleaned + "\n", encoding="utf-8")
        return True

    return False


class AgenticRuntime:
    """Hardened execution orchestrator for Craftly coding agents."""

    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend

    async def run_task(self, task: AgenticTask) -> AgenticExecutionReport:
        """Run the end-to-end task in an ephemeral sandbox workspace."""
        start_time = time.perf_counter()
        src_root = Path(task.workspace_dir).resolve()
        if not src_root.exists() or not src_root.is_dir():
            raise FileNotFoundError(f"Workspace directory not found: {src_root}")

        steps: list[AgenticStep] = []

        with tempfile.TemporaryDirectory(prefix="craftly_sandbox_") as tmp_sandbox:
            sandbox_root = Path(tmp_sandbox).resolve()
            shutil.copytree(src_root, sandbox_root, dirs_exist_ok=True)

            # Discover files in workspace
            workspace_files: list[str] = []
            for p in sorted(sandbox_root.rglob("*.py")):
                if not p.name.startswith("test_") and not p.name.startswith("."):
                    rel = p.relative_to(sandbox_root).as_posix()
                    content = p.read_text(encoding="utf-8", errors="replace")
                    workspace_files.append(f"File `{rel}`:\n```python\n{content.strip()}\n```")

            files_context = ("\n\nExisting Workspace Code:\n" + "\n\n".join(workspace_files)) if workspace_files else ""

            history_dialogue: list[str] = [
                f"Task: {task.instruction}\n"
                f"{files_context}\n\n"
                f"Available Tools: test, git_diff, shell\n"
                "Format tool calls as: <|tool_call|>{\"tool\": \"...\", \"command\": [...] }<|tool_end|>\n"
                "Format patches as: <|patch_start|>\n--- a/filename\n+++ b/filename\n...<|patch_end|>\n"
                "Or provide the complete corrected file/function code inside <|patch_start|> and <|patch_end|>.\n"
            ]

            status = "max_turns_exceeded"
            last_stdout = ""
            last_stderr = ""
            last_exit_code = None
            final_patch = None

            for turn in range(1, task.max_turns + 1):
                prompt = (
                    "System:\nYou are Craftly, an autonomous defensive cybersecurity and coding AI agent.\n\n"
                    "User:\n" + "\n".join(history_dialogue) + "\n\nAssistant:\n<|thought_start|>\n"
                )

                raw_generation = await self.backend.generate(
                    prompt,
                    temperature=0.2,
                    max_tokens=768,
                    stop_tokens=[TOOL_CALL_END, PATCH_END, "<|document_end|>"],
                )

                # Ensure delimiter completion if truncated at stop
                if TOOL_CALL_START in raw_generation and TOOL_CALL_END not in raw_generation:
                    raw_generation += TOOL_CALL_END
                if PATCH_START in raw_generation and PATCH_END not in raw_generation:
                    raw_generation += PATCH_END

                parsed = parse_agentic_output(raw_generation)
                step_record = AgenticStep(
                    turn=turn,
                    thought=parsed.thought,
                    tool_call=parsed.tool_call.model_dump() if parsed.tool_call else None,
                    patch=parsed.patch,
                    explanation=parsed.explanation,
                )

                # 1. Handle Tool Invocations
                if parsed.tool_call:
                    call = parsed.tool_call
                    tool_res = self._execute_safe_tool(call, sandbox_root)
                    step_record.tool_result = tool_res.model_dump()
                    history_dialogue.append(
                        f"Assistant:\n<|thought_start|>{parsed.thought or ''}<|thought_end|>\n"
                        f"<|tool_call|>{json.dumps(call.model_dump())}<|tool_end|>\n"
                        f"{format_tool_result(tool_res)}"
                    )

                # 2. Handle Patch (Unified Diff or Direct Code Replacement)
                elif parsed.patch:
                    final_patch = parsed.patch
                    applied = apply_patch_to_workspace(sandbox_root, parsed.patch)
                    if not applied:
                        history_dialogue.append(
                            f"Observation: Patch could not be applied cleanly. Check format."
                        )
                        steps.append(step_record)
                        continue

                    # Execute Verification Command in Sandbox
                    v_res = self._execute_safe_command(task.verification_command, sandbox_root)
                    last_stdout = v_res.stdout
                    last_stderr = v_res.stderr
                    last_exit_code = v_res.exit_code

                    if v_res.exit_code == 0:
                        status = "verified_pass"
                        steps.append(step_record)
                        break
                    else:
                        history_dialogue.append(
                            f"Observation: Patch applied, but verification failed (exit code {v_res.exit_code}).\n"
                            f"Stdout: {v_res.stdout[:2000]}\n"
                            f"Stderr: {v_res.stderr[:2000]}\n"
                            "Analyze the failure and refine your patch."
                        )

                else:
                    # Explanatory turn without tool call: check if verification passes as-is
                    v_res = self._execute_safe_command(task.verification_command, sandbox_root)
                    last_stdout = v_res.stdout
                    last_stderr = v_res.stderr
                    last_exit_code = v_res.exit_code
                    if v_res.exit_code == 0:
                        status = "verified_pass"
                        steps.append(step_record)
                        break

                steps.append(step_record)

            duration_ms = (time.perf_counter() - start_time) * 1000
            return AgenticExecutionReport(
                task_id=task.task_id,
                status=status,
                turns_taken=len(steps),
                patch=final_patch,
                verification_stdout=last_stdout,
                verification_stderr=last_stderr,
                verification_exit_code=last_exit_code,
                steps=steps,
                duration_ms=duration_ms,
            )

    def _execute_safe_tool(self, call: ToolCall, cwd: Path) -> ToolResult:
        """Execute allowlisted tool with root containment."""
        start = time.perf_counter()
        executable = call.command[0].lower()
        allowed = ALLOWED_TOOL_COMMANDS.get(call.tool, set())
        if executable not in allowed:
            return ToolResult(
                tool=call.tool,
                status="error",
                exit_code=1,
                stderr=f"Executable {executable!r} not permitted for tool {call.tool!r}",
            )

        return self._execute_safe_command(call.command, cwd, timeout_ms=call.timeout_ms)

    def _execute_safe_command(self, command: list[str], cwd: Path, timeout_ms: int = 30_000) -> ToolResult:
        """Run an allowlisted subprocess inside the sandboxed workspace."""
        start = time.perf_counter()
        # Resolve python executable to current python if requested
        resolved_cmd = list(command)
        if resolved_cmd[0].lower() in {"python", "python3", "python.exe"}:
            resolved_cmd[0] = sys.executable

        try:
            res = subprocess.run(  # nosec B603 - bounded allowlist, shell=False
                resolved_cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                timeout=timeout_ms / 1000.0,
                check=False,
            )
            duration = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool="command",
                status="ok" if res.returncode == 0 else "failed",
                exit_code=res.returncode,
                stdout=res.stdout,
                stderr=res.stderr,
                duration_ms=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool="command",
                status="timeout",
                exit_code=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "Command timed out",
                duration_ms=duration,
            )
        except Exception as err:
            duration = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool="command",
                status="error",
                exit_code=1,
                stderr=str(err),
                duration_ms=duration,
            )
