"""Defensive security scanner wrappers for Craftly.

The wrappers are intentionally conservative: missing external CLIs produce a
structured skipped result instead of a false pass or crash. Production gates can
decide whether skipped tools are acceptable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.shared.schemas import StrictModel


class SecurityScanResult(StrictModel):
    tool: str
    status: str
    finding_count: int = 0
    findings: list[dict[str, Any]] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: float = 0.0
    reason: str = ""


class SecurityScanner:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()

    def run_semgrep(self, *, timeout_ms: int = 120_000) -> SecurityScanResult:
        executable = shutil.which("semgrep")
        if executable is None:
            return SecurityScanResult(tool="semgrep", status="skipped", reason="semgrep CLI is not installed")
        started = time.perf_counter()
        command = [executable, "scan", "--config", "auto", "--json", "--quiet", "."]
        return self._run_json_scanner(
            tool="semgrep",
            command=command,
            timeout_ms=timeout_ms,
            finding_key="results",
            started=started,
        )

    def run_codeql(
        self,
        *,
        database: str | Path | None = None,
        suite: str = "security-and-quality",
        language: str | None = None,
        timeout_ms: int = 120_000,
    ) -> SecurityScanResult:
        executable = self._codeql_executable()
        if executable is None:
            return SecurityScanResult(tool="codeql", status="skipped", reason="codeql CLI is not installed")
        if database is None:
            return SecurityScanResult(tool="codeql", status="skipped", reason="codeql database path was not provided")
        database_path = Path(database).resolve()
        if not database_path.exists():
            return SecurityScanResult(tool="codeql", status="skipped", reason=f"codeql database not found: {database_path}")
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="craftly-codeql-results-") as output_dir:
            output_path = Path(output_dir) / "results.sarif"
            command = [
                executable,
                "database",
                "analyze",
                str(database_path),
                self._codeql_suite(language, suite),
                "--format=sarif-latest",
                f"--output={output_path}",
            ]
            try:
                completed = subprocess.run(  # nosec B603 - argv list, shell disabled.
                    command,
                    cwd=self.workspace,
                    capture_output=True,
                    text=True,
                    shell=False,
                    timeout=timeout_ms / 1000,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return SecurityScanResult(
                    tool="codeql",
                    status="timeout",
                    stdout=(exc.stdout if isinstance(exc.stdout, str) else "")[-20_000:],
                    stderr=(exc.stderr if isinstance(exc.stderr, str) else "")[-20_000:],
                    exit_code=None,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    reason="codeql timed out",
                )
            findings = self._read_sarif_findings(output_path)
        return SecurityScanResult(
            tool="codeql",
            status="failed" if findings else ("passed" if completed.returncode == 0 else "failed"),
            finding_count=len(findings),
            findings=findings,
            stdout=completed.stdout[-20_000:],
            stderr=completed.stderr[-20_000:],
            exit_code=completed.returncode,
            duration_ms=(time.perf_counter() - started) * 1000,
            reason="findings detected" if findings else "no findings detected",
        )

    def run_codeql_source(
        self,
        *,
        languages: list[str] | None = None,
        suite: str = "security-and-quality",
        timeout_ms: int = 300_000,
    ) -> SecurityScanResult:
        """Build temporary CodeQL databases from current source and analyze them."""

        executable = self._codeql_executable()
        if executable is None:
            return SecurityScanResult(tool="codeql", status="skipped", reason="codeql CLI is not installed")
        selected = languages or self._detect_codeql_languages()
        if not selected:
            return SecurityScanResult(
                tool="codeql",
                status="skipped",
                reason="no supported interpreted CodeQL language detected",
            )
        started = time.perf_counter()
        combined_findings: list[dict[str, Any]] = []
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        with tempfile.TemporaryDirectory(prefix="craftly-codeql-") as temp_dir:
            for language in selected:
                database = Path(temp_dir) / language
                create_command = [
                    executable,
                    "database",
                    "create",
                    str(database),
                    f"--language={language}",
                    f"--source-root={self.workspace}",
                    "--overwrite",
                ]
                if language in {"cpp", "csharp", "java-kotlin", "swift", "rust"}:
                    create_command.append("--build-mode=none")
                try:
                    created = subprocess.run(  # nosec B603 - fixed argv, shell disabled.
                        create_command,
                        cwd=self.workspace,
                        capture_output=True,
                        text=True,
                        shell=False,
                        timeout=timeout_ms / 1000,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    return SecurityScanResult(
                        tool="codeql",
                        status="timeout",
                        stdout=(exc.stdout if isinstance(exc.stdout, str) else "")[-20_000:],
                        stderr=(exc.stderr if isinstance(exc.stderr, str) else "")[-20_000:],
                        duration_ms=(time.perf_counter() - started) * 1000,
                        reason=f"CodeQL database creation timed out for {language}",
                    )
                stdout_parts.append(created.stdout[-10_000:])
                stderr_parts.append(created.stderr[-10_000:])
                if created.returncode != 0:
                    return SecurityScanResult(
                        tool="codeql",
                        status="failed",
                        stdout="\n".join(stdout_parts)[-20_000:],
                        stderr="\n".join(stderr_parts)[-20_000:],
                        exit_code=created.returncode,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        reason=f"CodeQL database creation failed for {language}",
                    )
                analyzed = self.run_codeql(
                    database=database,
                    suite=suite,
                    language=language,
                    timeout_ms=timeout_ms,
                )
                stdout_parts.append(analyzed.stdout)
                stderr_parts.append(analyzed.stderr)
                if analyzed.status in {"timeout", "skipped"}:
                    return analyzed
                combined_findings.extend(
                    [{**finding, "language": language} for finding in analyzed.findings]
                )
                if analyzed.exit_code not in {0, None} and not analyzed.findings:
                    return analyzed
        return SecurityScanResult(
            tool="codeql",
            status="failed" if combined_findings else "passed",
            finding_count=len(combined_findings),
            findings=combined_findings,
            stdout="\n".join(stdout_parts)[-20_000:],
            stderr="\n".join(stderr_parts)[-20_000:],
            exit_code=0,
            duration_ms=(time.perf_counter() - started) * 1000,
            reason="findings detected" if combined_findings else "no findings detected",
        )

    def _codeql_executable(self) -> str | None:
        configured = os.environ.get("CRAFTLY_CODEQL_BIN", "")
        if configured:
            path = Path(configured).expanduser().resolve()
            if path.is_file():
                return str(path)
        executable = shutil.which("codeql")
        if executable:
            return executable
        for candidate in [
            Path.home() / ".craftly" / "tools" / "codeql" / "codeql" / "codeql.exe",
            Path.home() / ".craftly" / "tools" / "codeql" / "codeql" / "codeql",
        ]:
            if candidate.is_file():
                return str(candidate.resolve())
        return None

    def _detect_codeql_languages(self) -> list[str]:
        suffixes = {path.suffix.lower() for path in self.workspace.rglob("*") if path.is_file()}
        selected: list[str] = []
        if suffixes.intersection({".py"}):
            selected.append("python")
        if suffixes.intersection({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}):
            selected.append("javascript-typescript")
        if suffixes.intersection({".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}):
            selected.append("cpp")
        if suffixes.intersection({".cs"}):
            selected.append("csharp")
        if suffixes.intersection({".java", ".kt", ".kts"}):
            selected.append("java-kotlin")
        if suffixes.intersection({".go"}):
            selected.append("go")
        if suffixes.intersection({".rb"}):
            selected.append("ruby")
        if suffixes.intersection({".swift"}):
            selected.append("swift")
        if suffixes.intersection({".rs"}):
            selected.append("rust")
        return selected

    @staticmethod
    def _codeql_suite(language: str | None, suite: str) -> str:
        if suite != "security-and-quality" or language is None:
            return suite
        package_names = {
            "python": ("python", "python"),
            "javascript-typescript": ("javascript", "javascript"),
            "cpp": ("cpp", "cpp"),
            "csharp": ("csharp", "csharp"),
            "java-kotlin": ("java", "java"),
            "go": ("go", "go"),
            "ruby": ("ruby", "ruby"),
            "swift": ("swift", "swift"),
            "rust": ("rust", "rust"),
        }
        package, prefix = package_names[language]
        return f"codeql/{package}-queries:codeql-suites/{prefix}-security-and-quality.qls"

    def _run_json_scanner(
        self,
        *,
        tool: str,
        command: list[str],
        timeout_ms: int,
        finding_key: str,
        started: float,
    ) -> SecurityScanResult:
        try:
            completed = subprocess.run(  # nosec B603 - argv list, shell disabled.
                command,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                shell=False,
                timeout=timeout_ms / 1000,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return SecurityScanResult(
                tool=tool,
                status="timeout",
                stdout=(exc.stdout if isinstance(exc.stdout, str) else "")[-20_000:],
                stderr=(exc.stderr if isinstance(exc.stderr, str) else "")[-20_000:],
                exit_code=None,
                duration_ms=(time.perf_counter() - started) * 1000,
                reason=f"{tool} timed out",
            )
        findings: list[dict[str, Any]] = []
        if completed.stdout.strip():
            try:
                payload = json.loads(completed.stdout)
                raw_findings = payload.get(finding_key, [])
                if isinstance(raw_findings, list):
                    findings = [self._compact_finding(item) for item in raw_findings]
            except json.JSONDecodeError:
                findings = []
        return SecurityScanResult(
            tool=tool,
            status="failed" if findings else ("passed" if completed.returncode == 0 else "failed"),
            finding_count=len(findings),
            findings=findings,
            stdout=completed.stdout[-20_000:],
            stderr=completed.stderr[-20_000:],
            exit_code=completed.returncode,
            duration_ms=(time.perf_counter() - started) * 1000,
            reason="findings detected" if findings else "no findings detected",
        )

    def _compact_finding(self, item: dict[str, Any]) -> dict[str, Any]:
        path = item.get("path") or item.get("extra", {}).get("path") or ""
        start = item.get("start") or {}
        extra = item.get("extra") or {}
        return {
            "rule_id": item.get("check_id") or item.get("rule_id") or "",
            "path": path,
            "line": start.get("line"),
            "message": extra.get("message") or item.get("message") or "",
            "severity": extra.get("severity") or "",
        }

    def _read_sarif_findings(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return []
        findings: list[dict[str, Any]] = []
        for run in payload.get("runs", []):
            rules = {
                rule.get("id"): rule.get("shortDescription", {}).get("text", "")
                for rule in run.get("tool", {}).get("driver", {}).get("rules", [])
            }
            for result in run.get("results", []):
                location = (result.get("locations") or [{}])[0].get("physicalLocation", {})
                artifact = location.get("artifactLocation", {})
                region = location.get("region", {})
                rule_id = result.get("ruleId") or ""
                findings.append(
                    {
                        "rule_id": rule_id,
                        "path": artifact.get("uri", ""),
                        "line": region.get("startLine"),
                        "message": result.get("message", {}).get("text") or rules.get(rule_id, ""),
                        "severity": result.get("level", ""),
                    }
                )
        return findings

