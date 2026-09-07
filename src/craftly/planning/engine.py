"""Native Intent & Planning Engine for the final Craftly architecture."""

from __future__ import annotations

import time
import json
from typing import Any

from pydantic import Field

from src.craftly.shared.schemas import StrictModel
from src.craftly.model_ops.backends import ModelBackend


class PlanningResult(StrictModel):
    run_id: str
    prompt: str
    expansion: dict[str, Any]
    task_graph_brief: str
    goal: str
    requirements: list[str]
    risks: list[str]
    success_criteria: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    created_at_unix: float = Field(default_factory=time.time)


class IntentPlanningEngine:
    def plan(self, prompt: str, *, run_id: str | None = None) -> PlanningResult:
        rid = run_id or f"craftly-plan-{time.time_ns()}"
        lowered = prompt.lower()
        intent = "security_review" if any(term in lowered for term in ["security", "vulnerability", "cve", "exploit"]) else "debug" if any(term in lowered for term in ["bug", "fix", "error", "fail"]) else "code_edit"
        requirements = [
            "understand the repository context",
            "retrieve relevant files and symbols",
            "propose a minimal patch",
            "run verification commands",
            "return evidence with the final answer",
        ]
        if intent == "security_review":
            requirements.append("run defensive security checks before accepting changes")
        risks = [
            "insufficient repository context",
            "patch may fail tests",
            "security regression if verification is skipped",
        ]
        success_criteria = [
            "context pack includes relevant source and tests",
            "patch preview is generated before apply",
            "configured tests pass",
            "verifier returns accept",
        ]
        expansion = {"intent": intent, "acceptance_tests": success_criteria, "source": "native-planner"}
        return PlanningResult(
            run_id=rid,
            prompt=prompt,
            expansion=expansion,
            task_graph_brief="understand -> planner -> retrieve_context -> edit -> (test || security_review || performance_review) -> critic_review -> verify -> summarize",
            goal=f"Complete Craftly request: {prompt}",
            requirements=requirements,
            risks=risks,
            success_criteria=success_criteria,
            confidence=0.82,
        )

    async def plan_structured(
        self,
        prompt: str,
        *,
        backend: ModelBackend,
        run_id: str | None = None,
        allow_dev_fallback: bool = False,
    ) -> PlanningResult:
        schema_prompt = (
            "Return only valid JSON for an Craftly coding-agent plan with keys: "
            "goal, requirements, risks, success_criteria, expansion. "
            "Do not write executable code in the planner output.\n\n"
            f"User request:\n{prompt}"
        )
        raw = await backend.generate(schema_prompt, temperature=0.0, max_tokens=900)
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("planner output must be a JSON object")
            requirements = self._required_string_list(payload, "requirements")
            risks = self._required_string_list(payload, "risks")
            success_criteria = self._required_string_list(payload, "success_criteria")
            goal = str(payload.get("goal") or "").strip()
            if not goal:
                raise ValueError("planner output missing goal")
            expansion = payload.get("expansion") if isinstance(payload.get("expansion"), dict) else {}
            expansion = {**expansion, "source": "structured-model-planner"}
            return PlanningResult(
                run_id=run_id or f"craftly-plan-{time.time_ns()}",
                prompt=prompt,
                expansion=expansion,
                task_graph_brief="understand -> planner -> retrieve_context -> edit -> (test || security_review || performance_review) -> critic_review -> verify -> summarize",
                goal=goal,
                requirements=requirements,
                risks=risks,
                success_criteria=success_criteria,
                confidence=0.88,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            if allow_dev_fallback:
                fallback = self.plan(prompt, run_id=run_id)
                fallback.expansion["source"] = "keyword-dev-fallback"
                fallback.risks.append(f"structured planner fallback used: {exc}")
                fallback.confidence = min(fallback.confidence, 0.62)
                return fallback
            raise ValueError(f"structured planner returned invalid JSON: {exc}") from exc

    @staticmethod
    def _required_string_list(payload: dict[str, Any], key: str) -> list[str]:
        values = payload.get(key)
        if not isinstance(values, list) or not values or not all(isinstance(item, str) and item.strip() for item in values):
            raise ValueError(f"planner output {key!r} must be a non-empty string list")
        return [item.strip() for item in values]

