"""Static architecture ownership and dependency integrity gate.

The gate is intentionally AST-based and deterministic. It does not import the
application under inspection, execute module code, or infer ownership from
file names alone.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from pydantic import Field

from src.craftly.shared.schemas import StrictModel

AUTHORITATIVE_MODULES = {
    "canonical_integrity": "src.craftly.shared.integrity",
    "configuration_contracts": "src.craftly.shared.config_contracts",
    "model_architecture_contract": "src.craftly.model_ops.foundation",
    "independent_review": "src.craftly.learning.training_data_gate",
    "production_release_decision": "src.craftly.deployment.production_qualification",
    "scientific_experiment_decision": "src.craftly.learning.ablation_runner",
    "tool_execution_policy": "src.craftly.tools.policy",
}
OWNED_CLASS_NAMES = {
    "model_architecture_contract": {
        "ScratchDecoderConfig",
        "TrainingBatchContract",
        "TokenizerContract",
        "ParallelismPlan",
    },
    "scientific_experiment_decision": {
        "ExperimentAuthority",
        "ExperimentManifest",
        "ExperimentDecision",
        "PromotionDecision",
    },
}
OWNED_FUNCTION_NAMES = {
    "canonical_integrity": {
        "canonical_json_bytes",
        "canonical_json_text",
        "sha256_file",
    },
    "independent_review": {"has_independent_review"},
    "tool_execution_policy": {"project_root"},
}


class FunctionLocation(StrictModel):
    module: str
    name: str
    line: int = Field(ge=1)


class DuplicateFunctionBody(StrictModel):
    digest: str
    locations: list[FunctionLocation]
    statement_count: int = Field(ge=1)
    line_count: int = Field(ge=1)


class ImportCycle(StrictModel):
    modules: list[str] = Field(min_length=2)


class OwnershipViolation(StrictModel):
    capability: str
    expected_module: str
    actual_module: str
    symbol: str
    line: int = Field(ge=1)


class ArchitectureIntegrityReport(StrictModel):
    status: str
    source_root: str
    parsed_modules: int = Field(ge=0)
    parse_errors: list[str] = Field(default_factory=list)
    ownership_violations: list[OwnershipViolation] = Field(default_factory=list)
    duplicate_function_bodies: list[DuplicateFunctionBody] = Field(default_factory=list)
    import_cycles: list[ImportCycle] = Field(default_factory=list)
    authoritative_modules: dict[str, str] = Field(
        default_factory=lambda: dict(AUTHORITATIVE_MODULES)
    )

    def write(self, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        target = root / "architecture_integrity_report.json"
        target.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


def _module_name(source_root: Path, path: Path) -> str:
    relative = path.relative_to(source_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("src", "craftly", *parts))


def _top_level_nodes(nodes: Iterable[ast.stmt]) -> Iterable[ast.stmt]:
    """Yield import-capable initialization nodes without entering functions."""
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.If):
            if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                continue
            yield from _top_level_nodes(node.body)
            yield from _top_level_nodes(node.orelse)
            continue
        if isinstance(node, ast.Try):
            yield from _top_level_nodes(node.body)
            for handler in node.handlers:
                yield from _top_level_nodes(handler.body)
            yield from _top_level_nodes(node.orelse)
            yield from _top_level_nodes(node.finalbody)
            continue
        yield node


def _import_targets(tree: ast.Module) -> set[str]:
    targets: set[str] = set()
    for node in _top_level_nodes(tree.body):
        if isinstance(node, ast.Import):
            targets.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("src.craftly")
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("src.craftly")
        ):
            targets.add(node.module)
    return targets


def _function_fingerprint(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, int, int] | None:
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    statement_count = sum(
        1
        for child in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(child, ast.stmt)
    )
    line_count = max(1, int(getattr(node, "end_lineno", node.lineno)) - node.lineno + 1)
    if statement_count < 5 or line_count < 8:
        return None
    payload = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
    return (
        hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        statement_count,
        line_count,
    )


def _canonical_cycle(cycle: list[str]) -> tuple[str, ...]:
    core = cycle[:-1] if len(cycle) > 1 and cycle[0] == cycle[-1] else cycle
    variants: list[tuple[str, ...]] = []
    for values in (core, list(reversed(core))):
        variants.extend(
            tuple(values[index:] + values[:index]) for index in range(len(values))
        )
    return min(variants)


def _find_cycles(graph: dict[str, set[str]]) -> list[ImportCycle]:
    cycles: set[tuple[str, ...]] = set()
    visiting: list[str] = []
    active: set[str] = set()
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visited:
            return
        active.add(module)
        visiting.append(module)
        for dependency in sorted(graph.get(module, set())):
            if dependency not in graph:
                continue
            if dependency in active:
                start = visiting.index(dependency)
                cycles.add(_canonical_cycle(visiting[start:] + [dependency]))
            elif dependency not in visited:
                visit(dependency)
        visiting.pop()
        active.remove(module)
        visited.add(module)

    for module in sorted(graph):
        visit(module)
    return [ImportCycle(modules=list(cycle)) for cycle in sorted(cycles)]


def run_architecture_integrity(
    *,
    repository_root: str | Path = ".",
) -> ArchitectureIntegrityReport:
    repo = Path(repository_root).resolve()
    source_root = repo / "src" / "craftly"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Craftly source root not found: {source_root}")

    trees: dict[str, ast.Module] = {}
    parse_errors: list[str] = []
    locations_by_digest: dict[str, list[FunctionLocation]] = defaultdict(list)
    fingerprint_metrics: dict[str, tuple[int, int]] = {}
    ownership_violations: list[OwnershipViolation] = []

    for path in sorted(source_root.rglob("*.py")):
        module = _module_name(source_root, path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            parse_errors.append(f"{path.relative_to(repo)}: {exc}")
            continue
        trees[module] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for capability, symbols in OWNED_CLASS_NAMES.items():
                    if (
                        node.name in symbols
                        and module != AUTHORITATIVE_MODULES[capability]
                    ):
                        ownership_violations.append(
                            OwnershipViolation(
                                capability=capability,
                                expected_module=AUTHORITATIVE_MODULES[capability],
                                actual_module=module,
                                symbol=node.name,
                                line=node.lineno,
                            )
                        )
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            fingerprint = _function_fingerprint(node)
            if fingerprint is not None:
                digest, statements, lines = fingerprint
                locations_by_digest[digest].append(
                    FunctionLocation(module=module, name=node.name, line=node.lineno)
                )
                fingerprint_metrics[digest] = statements, lines
            for capability, symbols in OWNED_FUNCTION_NAMES.items():
                if node.name in symbols and module != AUTHORITATIVE_MODULES[capability]:
                    ownership_violations.append(
                        OwnershipViolation(
                            capability=capability,
                            expected_module=AUTHORITATIVE_MODULES[capability],
                            actual_module=module,
                            symbol=node.name,
                            line=node.lineno,
                        )
                    )

    missing_authorities = {
        capability: module
        for capability, module in AUTHORITATIVE_MODULES.items()
        if module not in trees
    }
    for capability, module in sorted(missing_authorities.items()):
        ownership_violations.append(
            OwnershipViolation(
                capability=capability,
                expected_module=module,
                actual_module="<missing>",
                symbol="<module>",
                line=1,
            )
        )

    duplicates = []
    for digest, locations in sorted(locations_by_digest.items()):
        modules = {item.module for item in locations}
        if len(locations) < 2 or len(modules) < 2:
            continue
        statements, lines = fingerprint_metrics[digest]
        duplicates.append(
            DuplicateFunctionBody(
                digest=digest,
                locations=sorted(
                    locations, key=lambda item: (item.module, item.line, item.name)
                ),
                statement_count=statements,
                line_count=lines,
            )
        )

    graph = {
        module: {target for target in _import_targets(tree) if target != module}
        for module, tree in trees.items()
    }
    cycles = _find_cycles(graph)
    failed = bool(parse_errors or ownership_violations or duplicates or cycles)
    return ArchitectureIntegrityReport(
        status="failed" if failed else "passed",
        source_root=str(source_root),
        parsed_modules=len(trees),
        parse_errors=parse_errors,
        ownership_violations=ownership_violations,
        duplicate_function_bodies=duplicates,
        import_cycles=cycles,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Craftly static architecture integrity checks."
    )
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--output-dir", default="artifacts/craftly/architecture-integrity"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_architecture_integrity(repository_root=args.repository_root)
    report.write(args.output_dir)
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if report.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
