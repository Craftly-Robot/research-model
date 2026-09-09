"""Shared strict schemas for the consolidated Craftly runtime."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def report_json(report: StrictModel, *, indent: int = 2) -> str:
    """Serialize a Pydantic report model to a JSON string."""
    return report.model_dump_json(indent=indent)


def write_report(report: StrictModel, path: str | Path) -> Path:
    """Write a Pydantic report model to a JSON file. Returns the path written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report_json(report), encoding="utf-8")
    return target


def print_report(report: StrictModel) -> None:
    """Print a Pydantic report model as formatted JSON to stdout."""
    print(report_json(report))


class EvaluationGate(StrictModel):
    """Shared measured gate result used by checkpoint and regression evaluators."""

    name: str
    status: str
    reason: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class CraftlyRunRequest(StrictModel):
    prompt: str = Field(min_length=1)
    workspace: str = Field(default_factory=lambda: str(Path.cwd()))
    policy_mode: str = Field(default="strict", pattern="^(strict|development)$")
    agent_backend_mode: str = Field(default="auto", pattern="^(auto|active|mock)$")
    run_verifier: bool = True
    run_security: bool = True
    max_agent_nodes: int | None = Field(default=None, ge=1, le=12)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CraftlyRunReport(StrictModel):
    run_id: str
    status: str
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    prompt: str
    workspace: str
    final_answer: str
    route: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] | None = None
    memory: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    security: dict[str, Any] | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float
    created_at_unix: float = Field(default_factory=time.time)


class ModuleHealth(StrictModel):
    name: str
    ok: bool
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


class SystemHealth(StrictModel):
    ok: bool
    modules: list[ModuleHealth]
    created_at_unix: float = Field(default_factory=time.time)
