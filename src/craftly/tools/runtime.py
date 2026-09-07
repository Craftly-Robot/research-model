"""Compatibility tool runtime.

New code should import :class:`src.craftly.tools.policy.HardenedToolExecutor`.
This module keeps the historical public schemas stable and delegates execution
to the hardened policy path.
"""

from __future__ import annotations

from src.craftly.db import LocalStore
from src.craftly.tools.policy import (
    ExecutionRequest,
    ExecutionResult,
    HardenedToolExecutor,
    SandboxPolicy,
    SandboxRequest,
    ToolExecuteRequest,
    ToolExecuteResponse,
    project_root,
)


def validate_command_policy(request: ToolExecuteRequest) -> None:
    HardenedToolExecutor().validate_tool_shape(request)


class ToolRuntime:
    def __init__(self, store: LocalStore | None = None) -> None:
        self.store = store or LocalStore()

    def execute(self, request: ToolExecuteRequest) -> ToolExecuteResponse:
        return HardenedToolExecutor(self.store).execute(request)


class SandboxEngine:
    async def run(self, request: ExecutionRequest) -> ExecutionResult:
        result = ToolRuntime().execute(request)
        return ExecutionResult.model_validate(result.model_dump())

