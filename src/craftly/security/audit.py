"""Production security audit for Craftly source and deployment assets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.deployment.k8s_validate import validate_manifests
from src.craftly.shared.config_contracts import load_security_audit_contract
from src.craftly.shared.schemas import StrictModel


SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*=\s*['\"][^'\"]{12,}['\"]")
SSRF_PATTERN = re.compile(r"(?i)(requests\.(get|post|put|delete)|httpx\.(get|post|put|delete)|urlopen)\s*\(")
PATH_TRAVERSAL_PATTERN = re.compile(r"(?i)(open|Path)\s*\([^)]*(request|args|params|user_input|filename)")
DANGEROUS_SUBPROCESS_PATTERN = re.compile(r"(?i)(shell\s*=\s*True|os\.system|subprocess\.(call|run|Popen)\([^)]*\+)")
EXECUTABLE_SINK_PATTERN = re.compile(r"(?i)\b(subprocess\.(run|call|Popen)|os\.system|eval\s*\(|exec\s*\(|child_process\.exec)")


class SecurityFinding(StrictModel):
    severity: str
    check: str
    file: str
    line: int = 0
    message: str
    evidence: str = ""


class SecurityAuditReport(StrictModel):
    status: str
    root: str
    findings: list[SecurityFinding]
    dependency_warnings: list[str]
    external_scanners: dict[str, Any] = Field(default_factory=dict)
    scanner_install_plan: dict[str, Any] = Field(default_factory=dict)
    bandit: dict[str, Any] | None = None
    k8s: dict[str, Any] | None = None
    created_at_unix: float = Field(default_factory=time.time)

    def write(self, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        target = root / "security_audit_report.json"
        target.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
        write_markdown(self, root / "security_audit_report.md")
        return target


def _load_audit_excludes(root: Path) -> dict[str, dict[str, Any]]:
    config_path = root / "config" / "security_audit_excludes.json"
    if not config_path.exists():
        return {}
    payload = load_security_audit_contract(config_path).model_dump()
    excludes: dict[str, dict[str, Any]] = {}
    for item in payload.get("excludes", []):
        path = str(item.get("path") or "").replace("\\", "/")
        reason = str(item.get("reason") or "").strip()
        risk_category = str(item.get("risk_category") or "").strip()
        if not path or path.startswith("../") or "/../" in f"/{path}/":
            raise ValueError(f"invalid audit exclude path: {path!r}")
        if not reason or not risk_category:
            raise ValueError(f"audit exclude requires reason and risk_category: {path}")
        excludes[path] = item
    return excludes


def _validate_audit_excludes(root: Path, excludes: dict[str, dict[str, Any]]) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    for relative, item in excludes.items():
        target = (root / relative).resolve()
        if not target.exists():
            findings.append(
                SecurityFinding(
                    severity="fail",
                    check="audit_exclude_missing",
                    file=relative,
                    message="security audit exclude points to a missing file",
                )
            )
            continue
        if bool(item.get("allow_executable_sinks")):
            continue
        for line_number, line in enumerate(target.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if EXECUTABLE_SINK_PATTERN.search(line):
                findings.append(
                    SecurityFinding(
                        severity="fail",
                        check="audit_exclude_executable_sink",
                        file=relative,
                        line=line_number,
                        message="excluded file contains executable sink without explicit approval",
                        evidence=line.strip()[:240],
                    )
                )
    return findings


def _iter_source_files(root: Path, excluded_paths: set[str] | None = None) -> list[Path]:
    suffixes = {".py", ".js", ".ts", ".tsx", ".go", ".rs", ".java", ".yaml", ".yml", ".toml", ".env", ".txt"}
    excluded = {".git", "__pycache__", "artifacts", ".venv", "node_modules", "tests"}
    excluded_paths = excluded_paths or set()
    paths = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(part in excluded for part in path.parts):
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        if relative.as_posix() in excluded_paths:
            continue
        paths.append(path)
    return paths


def _scan_patterns(root: Path, excluded_paths: set[str] | None = None) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    for path in _iter_source_files(root, excluded_paths=excluded_paths):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if "replace-with" in stripped or "nosec" in stripped or "pragma: allowlist secret" in stripped:
                continue
            checks = [
                ("secret_pattern", SECRET_PATTERN, "fail", "possible hardcoded secret"),
                ("ssrf_sink", SSRF_PATTERN, "warn", "network request sink requires allowlist validation"),
                ("path_traversal_sink", PATH_TRAVERSAL_PATTERN, "warn", "file path sink may need canonical path validation"),
                ("dangerous_process", DANGEROUS_SUBPROCESS_PATTERN, "warn", "process execution sink requires strict argument handling"),
            ]
            for check, pattern, severity, message in checks:
                if pattern.search(stripped):
                    findings.append(
                        SecurityFinding(
                            severity=severity,
                            check=check,
                            file=str(path),
                            line=line_number,
                            message=message,
                            evidence=stripped[:240],
                        )
                    )
    return findings


def _dependency_warnings(root: Path) -> list[str]:
    warnings: list[str] = []
    for filename in ["requirements.txt", "requirements-local-dev.txt", "requirements-linux-gpu.txt"]:
        path = root / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if all(operator not in stripped for operator in ["==", ">=", "~=", "<="]):
                warnings.append(f"{filename}: dependency is not version bounded: {stripped}")
    return warnings


def _resolve_python_scanner(executable: str, module: str) -> tuple[list[str] | None, str | None]:
    resolved = shutil.which(executable)
    if resolved:
        return [resolved], None
    scripts_candidate = Path(sys.executable).resolve().parent / "Scripts" / f"{executable}.exe"
    if scripts_candidate.exists():
        return [str(scripts_candidate)], None
    if importlib.util.find_spec(module) is not None:
        return [sys.executable, "-m", module], None
    return None, f"{executable} executable or python module {module!r} is not installed"


def _resolve_codeql() -> tuple[list[str] | None, str | None]:
    env_path = Path(str(Path.home())) / "__missing__"
    if "CRAFTLY_CODEQL_BIN" in os.environ:
        env_path = Path(os.environ["CRAFTLY_CODEQL_BIN"]).expanduser()
        if env_path.exists():
            return [str(env_path)], None
    resolved = shutil.which("codeql")
    if resolved:
        return [resolved], None
    local_candidates = [
        Path.home() / ".craftly" / "tools" / "codeql" / "codeql" / "codeql.exe",
        Path.home() / ".craftly" / "tools" / "codeql" / "codeql" / "codeql",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return [str(candidate)], None
    return None, f"codeql executable is not installed; checked PATH, CRAFTLY_CODEQL_BIN={env_path}, and {local_candidates[0]}"


def _scanner_timeout_seconds(scanner: str, default: int) -> int:
    variable = f"CRAFTLY_{scanner.upper().replace('-', '_')}_TIMEOUT_SECONDS"
    raw = os.environ.get(variable, str(default)).strip()
    try:
        timeout = int(raw)
    except ValueError as exc:
        raise ValueError(f"{variable} must be an integer number of seconds") from exc
    if not 30 <= timeout <= 3_600:
        raise ValueError(f"{variable} must be between 30 and 3600 seconds")
    return timeout


def _scanner_positive_int(scanner: str, setting: str, default: int, *, minimum: int, maximum: int) -> int:
    variable = f"CRAFTLY_{scanner.upper().replace('-', '_')}_{setting.upper()}"
    raw = os.environ.get(variable, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{variable} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{variable} must be between {minimum} and {maximum}")
    return value


def _run_bandit(root: Path) -> dict[str, Any] | None:
    base_command, missing = _resolve_python_scanner("bandit", "bandit")
    if missing:
        return {"status": "skipped", "reason": missing}
    command = [*base_command, "-q", "-r", "src/craftly", "-f", "json"]
    timeout = _scanner_timeout_seconds("bandit", 600)
    try:
        completed = subprocess.run(  # nosec B603
            command,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": f"Bandit exceeded {timeout} seconds"}
    if completed.returncode not in {0, 1}:
        return {"status": "skipped", "reason": completed.stderr[-1000:] or completed.stdout[-1000:]}
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return {"status": "failed", "reason": "bandit returned invalid JSON", "stderr": completed.stderr[-1000:]}
    results = payload.get("results", [])
    blocking = [item for item in results if item.get("issue_severity") in {"MEDIUM", "HIGH"}]
    return {
        "status": "passed" if not blocking else "failed",
        "issue_count": len(results),
        "blocking_issue_count": len(blocking),
        "blocking_findings": [
            {
                "file": str(item.get("filename") or ""),
                "line": int(item.get("line_number") or 0),
                "rule": str(item.get("test_id") or "unknown"),
                "severity": str(item.get("issue_severity") or "unknown"),
                "message": str(item.get("issue_text") or "")[:500],
            }
            for item in blocking
        ],
        "metrics": payload.get("metrics", {}),
    }


def _run_semgrep(root: Path) -> dict[str, Any]:
    base_command, missing = _resolve_python_scanner("semgrep", "semgrep")
    completed: subprocess.CompletedProcess[str] | None = None
    backend = "local"
    if not missing:
        command = [*base_command, "scan", "--config", "auto", "--json", "src/craftly"]
        timeout = _scanner_timeout_seconds("semgrep", 900)
        try:
            completed = subprocess.run(  # nosec B603
                command,
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "failed", "backend": backend, "reason": f"Semgrep exceeded {timeout} seconds"}
    if completed is None or completed.returncode not in {0, 1}:
        docker = shutil.which("docker")
        image = os.environ.get(
            "CRAFTLY_SEMGREP_IMAGE",
            "semgrep/semgrep@sha256:2b33f46ba66cf8cc2ad59ccfa7d22951fd00c632c38f1339e84ec8e6e641a942",
        )
        if docker and re.fullmatch(r"semgrep/semgrep@sha256:[0-9a-f]{64}", image):
            docker_command = [
                docker,
                "run",
                "--rm",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=256",
                "--memory=2g",
                "--cpus=2",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=256m",  # nosec B108 - isolated container tmpfs, not a host temp path
                "--tmpfs",
                "/root:rw,noexec,nosuid,size=256m",
                "--env",
                "SEMGREP_ENABLE_VERSION_CHECK=0",
                "--volume",
                f"{root}:/src:ro",
                "--workdir",
                "/src",
                image,
                "semgrep",
                "scan",
                "--metrics",
                "off",
                "--config",
                "p/python",
                "--json",
                "src/craftly",
            ]
            # Docker is resolved from PATH, the image is digest-pinned, root is canonical, and shell execution is disabled.
            # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-tainted-env-args.dangerous-subprocess-use-tainted-env-args
            timeout = _scanner_timeout_seconds("semgrep", 900)
            try:
                completed = subprocess.run(  # nosec B603
                    docker_command,  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-tainted-env-args.dangerous-subprocess-use-tainted-env-args
                    cwd=root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return {
                    "status": "failed",
                    "backend": "docker_pinned",
                    "reason": f"Semgrep Docker scan exceeded {timeout} seconds",
                }
            backend = "docker_pinned"
        elif completed is None:
            return {"status": "skipped", "reason": missing or "Semgrep is unavailable"}
    assert completed is not None
    if completed.returncode not in {0, 1}:
        return {
            "status": "failed",
            "backend": backend,
            "reason": completed.stderr[-2000:] or completed.stdout[-2000:],
        }
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return {"status": "failed", "reason": "semgrep returned invalid JSON", "stderr": completed.stderr[-1000:]}
    findings = payload.get("results", [])
    error_count = sum(1 for item in findings if item.get("extra", {}).get("severity") == "ERROR")
    warning_count = sum(1 for item in findings if item.get("extra", {}).get("severity") == "WARNING")
    return {
        "status": "passed" if error_count == 0 else "failed",
        "backend": backend,
        "issue_count": len(findings),
        "error_count": error_count,
        "warning_count": warning_count,
        "blocking_findings": [
            {
                "file": str(item.get("path") or ""),
                "line": int((item.get("start") or {}).get("line") or 0),
                "rule": str(item.get("check_id") or "unknown"),
                "message": str((item.get("extra") or {}).get("message") or "")[:500],
            }
            for item in findings
            if (item.get("extra") or {}).get("severity") == "ERROR"
        ],
    }


def _run_codeql(root: Path) -> dict[str, Any]:
    base_command, missing = _resolve_codeql()
    if missing:
        return {"status": "skipped", "reason": missing}
    source_root = root / "src"
    if not source_root.is_dir():
        return {"status": "failed", "phase": "source_validation", "reason": f"missing source directory: {source_root}"}
    output = root / "artifacts" / "craftly" / "codeql-results.sarif"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="craftly-codeql-")).resolve()
    database = temporary_root / "database"
    # Unbounded CodeQL workers can exhaust an 8-16 GiB developer or CI host
    # before the operating system can report a useful error. These defaults
    # preserve the full security-extended query suite while bounding resource
    # use; larger runners can opt in to greater parallelism explicitly.
    threads = _scanner_positive_int("codeql", "threads", 1, minimum=1, maximum=64)
    # Keep the local default viable on an 8 GiB developer/CI worker. Larger
    # runners should raise CRAFTLY_CODEQL_RAM_MB explicitly; CodeQL fails
    # closed if the bounded process cannot complete.
    ram_mb = _scanner_positive_int("codeql", "ram_mb", 2048, minimum=1024, maximum=524_288)
    timeout = _scanner_timeout_seconds("codeql", 3600)
    try:
        create_command = [
            *base_command,
            "database",
            "create",
            str(database),
            "--language=python",
            "--build-mode=none",
            f"--source-root={source_root}",
            f"--threads={threads}",
            f"--ram={ram_mb}",
        ]
        try:
            created = subprocess.run(  # nosec B603
                create_command,
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "phase": "database_create",
                "reason": f"CodeQL database creation exceeded {timeout} seconds",
                "threads": threads,
                "ram_mb": ram_mb,
            }
        if created.returncode != 0:
            return {
                "status": "failed",
                "phase": "database_create",
                "returncode": created.returncode,
                "ephemeral_database": True,
                "threads": threads,
                "ram_mb": ram_mb,
                "stderr": created.stderr[-4000:],
            }
        command = [
            *base_command,
            "database",
            "analyze",
            str(database),
            "codeql/python-queries:codeql-suites/python-security-extended.qls",
            "--format=sarif-latest",
            f"--output={output}",
            f"--threads={threads}",
            f"--ram={ram_mb}",
            "--rerun",
        ]
        try:
            completed = subprocess.run(  # nosec B603
                command,
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "phase": "database_analyze",
                "reason": f"CodeQL analysis exceeded {timeout} seconds",
                "threads": threads,
                "ram_mb": ram_mb,
            }
        blocking = []
        if completed.returncode == 0 and output.is_file():
            try:
                sarif = json.loads(output.read_text(encoding="utf-8-sig"))
                blocking = [result for run in sarif.get("runs", []) for result in run.get("results", [])]
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                return {"status": "failed", "phase": "sarif_parse", "reason": str(exc), "output": str(output)}
        return {
            "status": "passed" if completed.returncode == 0 and not blocking else "failed",
            "phase": "database_analyze",
            "returncode": completed.returncode,
            "blocking_issue_count": len(blocking),
            "blocking_rules": sorted({str(item.get("ruleId") or "unknown") for item in blocking}),
            "source_root": str(source_root),
            "query_suite": "python-security-extended.qls",
            "ephemeral_database": True,
            "threads": threads,
            "ram_mb": ram_mb,
            "output": str(output),
            "reason": (
                ""
                if completed.returncode == 0
                else "CodeQL terminated abnormally; inspect stderr and lower CRAFTLY_CODEQL_THREADS "
                "or CRAFTLY_CODEQL_RAM_MB if the host is memory constrained"
            ),
            "stderr": completed.stderr[-4000:],
        }
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _run_pip_audit(root: Path) -> dict[str, Any]:
    base_command, missing = _resolve_python_scanner("pip-audit", "pip_audit")
    if missing:
        return {"status": "skipped", "reason": missing}
    command = [*base_command, "--format", "json"]
    timeout = _scanner_timeout_seconds("pip_audit", 600)
    try:
        completed = subprocess.run(  # nosec B603
            command,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": f"pip-audit exceeded {timeout} seconds"}
    if completed.returncode not in {0, 1}:
        return {"status": "failed", "reason": completed.stderr[-2000:] or completed.stdout[-2000:]}
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return {"status": "failed", "reason": "pip-audit returned invalid JSON", "stderr": completed.stderr[-1000:]}
    vulnerabilities = payload.get("vulnerabilities", [])
    return {"status": "passed" if not vulnerabilities else "failed", "vulnerability_count": len(vulnerabilities)}


def scanner_install_plan() -> dict[str, Any]:
    return {
        "python_tools": {
            "command": ["python", "-m", "pip", "install", "--upgrade", "bandit", "semgrep", "pip-audit"],
            "tools": ["bandit", "semgrep", "pip-audit"],
        },
        "codeql": {
            "windows": ["winget", "install", "--id", "GitHub.CodeQL"],
            "manual_windows_zip": "https://github.com/github/codeql-cli-binaries/releases/latest/download/codeql-win64.zip",
            "local_install_dir": str(Path.home() / ".craftly" / "tools" / "codeql"),
            "env_override": "CRAFTLY_CODEQL_BIN",
            "linux_note": "Install the official CodeQL CLI bundle from GitHub and add the codeql executable to PATH.",
            "required_after_install": ["codeql database create artifacts/craftly/codeql-db --language=python --source-root=."],
        },
        "strict_audit_command": [
            "python",
            "-m",
            "src.craftly.security.audit",
            "--strict-external-tools",
            "--output-dir",
            "artifacts/craftly/security-audit",
        ],
    }


def run_security_audit(
    *,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
    run_bandit: bool = True,
    validate_k8s: bool = True,
    run_semgrep: bool = True,
    run_codeql: bool = True,
    run_pip_audit: bool = True,
    strict_external_tools: bool = False,
) -> SecurityAuditReport:
    project_root = Path(root).resolve()
    audit_excludes = _load_audit_excludes(project_root)
    findings = _validate_audit_excludes(project_root, audit_excludes)
    findings.extend(_scan_patterns(project_root, excluded_paths=set(audit_excludes)))
    dependency_warnings = _dependency_warnings(project_root)
    bandit_report = _run_bandit(project_root) if run_bandit else None
    external_scanners: dict[str, Any] = {}
    if run_semgrep:
        external_scanners["semgrep"] = _run_semgrep(project_root)
    if run_codeql:
        external_scanners["codeql"] = _run_codeql(project_root)
    if run_pip_audit:
        external_scanners["pip_audit"] = _run_pip_audit(project_root)
    k8s_report = None
    if validate_k8s:
        manifests = sorted((project_root / "deploy" / "k8s").glob("*.yaml"))
        if manifests:
            k8s_report = validate_manifests(manifests).model_dump()
    failed = any(item.severity == "fail" for item in findings)
    failed = failed or bool(dependency_warnings)
    failed = failed or bool(bandit_report and bandit_report.get("status") == "failed")
    failed = failed or any(report.get("status") == "failed" for report in external_scanners.values())
    if strict_external_tools:
        failed = failed or bool(bandit_report and bandit_report.get("status") == "skipped")
        failed = failed or any(report.get("status") == "skipped" for report in external_scanners.values())
    failed = failed or bool(k8s_report and k8s_report.get("status") == "failed")
    report = SecurityAuditReport(
        status="failed" if failed else "passed",
        root=str(project_root),
        findings=findings,
        dependency_warnings=dependency_warnings,
        external_scanners=external_scanners,
        scanner_install_plan=scanner_install_plan(),
        bandit=bandit_report,
        k8s=k8s_report,
    )
    if output_dir:
        report.write(output_dir)
    return report


def write_markdown(report: SecurityAuditReport, path: str | Path) -> Path:
    target = Path(path)
    lines = [
        "# Craftly Security Audit Report",
        "",
        f"- status: {report.status}",
        f"- findings: {len(report.findings)}",
        f"- dependency_warnings: {len(report.dependency_warnings)}",
        "",
        "| severity | check | file | line | message |",
        "|---|---|---|---:|---|",
    ]
    for finding in report.findings:
        lines.append(f"| {finding.severity} | {finding.check} | {finding.file} | {finding.line} | {finding.message} |")
    if report.dependency_warnings:
        lines.extend(["", "## Dependency Warnings", ""])
        for warning in report.dependency_warnings:
            lines.append(f"- {warning}")
    if report.external_scanners:
        lines.extend(["", "## External Scanners", "", "| scanner | status | detail |", "|---|---|---|"])
        for name, payload in report.external_scanners.items():
            detail = payload.get("reason") or payload.get("issue_count") or payload.get("vulnerability_count") or ""
            lines.append(f"| {name} | {payload.get('status')} | {detail} |")
    lines.extend(["", "## Scanner Install Plan", ""])
    lines.append("```powershell")
    lines.append(" ".join(report.scanner_install_plan["python_tools"]["command"]))
    lines.append(" ".join(report.scanner_install_plan["codeql"]["windows"]))
    lines.append(" ".join(report.scanner_install_plan["strict_audit_command"]))
    lines.append("```")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Craftly production security audit.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="artifacts/craftly/security-audit")
    parser.add_argument("--no-bandit", action="store_true")
    parser.add_argument("--no-k8s", action="store_true")
    parser.add_argument("--no-semgrep", action="store_true")
    parser.add_argument("--no-codeql", action="store_true")
    parser.add_argument("--no-pip-audit", action="store_true")
    parser.add_argument("--strict-external-tools", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_security_audit(
        root=args.root,
        output_dir=args.output_dir,
        run_bandit=not args.no_bandit,
        validate_k8s=not args.no_k8s,
        run_semgrep=not args.no_semgrep,
        run_codeql=not args.no_codeql,
        run_pip_audit=not args.no_pip_audit,
        strict_external_tools=args.strict_external_tools,
    )
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
    if report.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

