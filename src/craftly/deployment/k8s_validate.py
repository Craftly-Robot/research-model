"""Kubernetes manifest validator for Craftly production deployments."""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.shared.schemas import StrictModel


class K8sValidationIssue(StrictModel):
    severity: str
    code: str
    message: str
    file: str
    resource: str = ""


class K8sValidationReport(StrictModel):
    status: str
    files: list[str]
    resources: dict[str, int]
    issue_count: int
    issues: list[K8sValidationIssue]
    kubectl_dry_run: dict[str, Any] | None = None
    created_at_unix: float = Field(default_factory=time.time)

    def write(self, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        target = root / "k8s_validation_report.json"
        target.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
        write_markdown(self, root / "k8s_validation_report.md")
        return target


def _load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    import yaml

    return [
        doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if isinstance(doc, dict)
    ]


def _resource_name(doc: dict[str, Any]) -> str:
    return f"{doc.get('kind', 'Unknown')}/{doc.get('metadata', {}).get('name', 'unnamed')}"


def _pod_template(doc: dict[str, Any]) -> dict[str, Any] | None:
    kind = doc.get("kind")
    spec = doc.get("spec", {})
    if kind in {"Deployment", "StatefulSet", "DaemonSet"}:
        return spec.get("template", {}).get("spec", {})
    if kind == "Job":
        return spec.get("template", {}).get("spec", {})
    if kind == "Pod":
        return spec
    return None


def _container_issues(
    path: Path, resource: str, pod_spec: dict[str, Any]
) -> list[K8sValidationIssue]:
    issues: list[K8sValidationIssue] = []
    for container in pod_spec.get("containers", []):
        name = str(container.get("name", "container"))
        security_context = (
            container.get("securityContext", {})
            if isinstance(container.get("securityContext"), dict)
            else {}
        )
        if security_context.get("privileged") is True:
            issues.append(
                K8sValidationIssue(
                    severity="fail",
                    code="privileged_container",
                    message=f"{name} is privileged",
                    file=str(path),
                    resource=resource,
                )
            )
        if security_context.get("allowPrivilegeEscalation") is True:
            issues.append(
                K8sValidationIssue(
                    severity="fail",
                    code="privilege_escalation",
                    message=f"{name} allows privilege escalation",
                    file=str(path),
                    resource=resource,
                )
            )
        resources = (
            container.get("resources", {}) if isinstance(container.get("resources"), dict) else {}
        )
        if "requests" not in resources or "limits" not in resources:
            issues.append(
                K8sValidationIssue(
                    severity="warn",
                    code="missing_resources",
                    message=f"{name} has no complete resource requests/limits",
                    file=str(path),
                    resource=resource,
                )
            )
        if doc_requires_probe(resource) and (
            "readinessProbe" not in container or "livenessProbe" not in container
        ):
            issues.append(
                K8sValidationIssue(
                    severity="warn",
                    code="missing_probe",
                    message=f"{name} lacks readiness/liveness probes",
                    file=str(path),
                    resource=resource,
                )
            )
        for env in container.get("env", []):
            if not isinstance(env, dict):
                continue
            value = str(env.get("value", ""))
            name_key = str(env.get("name", "")).lower()
            is_secret_like = any(
                secret_word in name_key for secret_word in ("secret", "password", "token", "key")
            )
            is_control_flag = name_key in {
                "craftly_allow_token_issue",
                "craftly_auth_enabled",
                "craftly_quota_enabled",
            }
            if is_secret_like and not is_control_flag and value and "replace-with" not in value:
                issues.append(
                    K8sValidationIssue(
                        severity="fail",
                        code="inline_secret",
                        message=f"{name} has inline secret-like env var {env.get('name')}",
                        file=str(path),
                        resource=resource,
                    )
                )
    return issues


def doc_requires_probe(resource: str) -> bool:
    return resource.startswith("Deployment/craftly-api")


def validate_manifests(
    paths: list[str | Path], *, kubectl_dry_run: bool = False
) -> K8sValidationReport:
    issues: list[K8sValidationIssue] = []
    resources: dict[str, int] = {}
    files = [str(path) for path in paths]
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            issues.append(
                K8sValidationIssue(
                    severity="fail",
                    code="missing_file",
                    message="manifest file missing",
                    file=str(path),
                )
            )
            continue
        for doc in _load_yaml_documents(path):
            kind = str(doc.get("kind", "Unknown"))
            resources[kind] = resources.get(kind, 0) + 1
            resource = _resource_name(doc)
            pod_spec = _pod_template(doc)
            if pod_spec is not None:
                issues.extend(_container_issues(path, resource, pod_spec))
            if (
                kind in {"Deployment", "StatefulSet"}
                and int(doc.get("spec", {}).get("replicas", 1)) < 1
            ):
                issues.append(
                    K8sValidationIssue(
                        severity="fail",
                        code="zero_replicas",
                        message="workload has zero replicas",
                        file=str(path),
                        resource=resource,
                    )
                )
            if kind == "Secret":
                text = path.read_text(encoding="utf-8")
                if "replace-with" in text:
                    issues.append(
                        K8sValidationIssue(
                            severity="warn",
                            code="example_secret",
                            message="example secret contains placeholder values",
                            file=str(path),
                            resource=resource,
                        )
                    )
    required = {
        "Deployment",
        "StatefulSet",
        "Service",
        "PersistentVolumeClaim",
        "HorizontalPodAutoscaler",
        "NetworkPolicy",
    }
    missing = sorted(required - set(resources))
    for kind in missing:
        issues.append(
            K8sValidationIssue(
                severity="fail",
                code="missing_required_kind",
                message=f"missing required kind {kind}",
                file=",".join(files),
            )
        )
    dry_run_result: dict[str, Any] | None = None
    if kubectl_dry_run:
        command = ["kubectl", "apply", "--dry-run=server", "-f", ",".join(files)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)  # nosec B603
        dry_run_result = {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
        if completed.returncode != 0:
            issues.append(
                K8sValidationIssue(
                    severity="fail",
                    code="kubectl_dry_run_failed",
                    message=completed.stderr[-1000:] or completed.stdout[-1000:],
                    file=",".join(files),
                )
            )
    status = "failed" if any(issue.severity == "fail" for issue in issues) else "passed"
    return K8sValidationReport(
        status=status,
        files=files,
        resources=resources,
        issue_count=len(issues),
        issues=issues,
        kubectl_dry_run=dry_run_result,
    )


def write_markdown(report: K8sValidationReport, path: str | Path) -> Path:
    target = Path(path)
    lines = [
        "# Craftly Kubernetes Validation Report",
        "",
        f"- status: {report.status}",
        f"- resources: {report.resources}",
        "",
        "| severity | code | resource | file | message |",
        "|---|---|---|---|---|",
    ]
    for issue in report.issues:
        lines.append(
            f"| {issue.severity} | {issue.code} | {issue.resource} | {issue.file} | {issue.message} |"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Craftly Kubernetes manifests.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=[str(path) for path in sorted(Path("deploy/k8s").glob("*.yaml"))],
    )
    parser.add_argument("--output-dir", default="artifacts/craftly/k8s-validation")
    parser.add_argument("--kubectl-dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = validate_manifests(args.files, kubectl_dry_run=args.kubectl_dry_run)
    report.write(args.output_dir)
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
    if report.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
