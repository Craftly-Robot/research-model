"""One-command benchmark pack runner for Craftly coding/security evaluation."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import time
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlparse

import httpx
from pydantic import Field, field_validator, model_validator

from src.craftly.evaluation.benchmarks import built_in_security_tasks
from src.craftly.evaluation.benchmark_suites import BenchmarkSuiteSpec, BenchmarkSuitesReport, run_benchmark_suites
from src.craftly.learning.benchmark_contamination_filter import build_protected_fingerprint_index
from src.craftly.shared.schemas import StrictModel
from src.craftly.shared.integrity import sha256_file as _sha256_file


class BenchmarkPackConfig(StrictModel):
    human_eval_path: str | None = None
    mbpp_path: str | None = None
    swe_bench_path: str | None = None
    cyberseceval_path: str | None = None
    custom_security_path: str | None = None
    strict: bool = True
    production: bool = False
    min_human_eval_tasks: int = 164
    min_mbpp_tasks: int = 374
    min_swe_bench_tasks: int = 1
    min_cyberseceval_tasks: int = 1


class BenchmarkPackReport(StrictModel):
    status: str
    strict: bool
    required_suites: list[str]
    optional_suites: list[str]
    suite_report: dict
    recommendations: list[str] = Field(default_factory=list)


class BenchmarkMaterializationReport(StrictModel):
    status: str
    output_dir: str
    files: dict[str, str]
    rows: dict[str, int]
    sources: dict[str, str]
    revisions: dict[str, str]
    licenses: dict[str, str]
    sha256: dict[str, str]
    train_policy: dict[str, Literal["eval_holdout"]]
    created_at_unix: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def validate_artifact_bindings(self) -> "BenchmarkMaterializationReport":
        expected_keys = set(self.files)
        for name, values in {
            "rows": self.rows,
            "sources": self.sources,
            "revisions": self.revisions,
            "licenses": self.licenses,
            "sha256": self.sha256,
            "train_policy": self.train_policy,
        }.items():
            if set(values) != expected_keys:
                raise ValueError(f"materialization report {name} keys do not match files")
        for digest in self.sha256.values():
            normalized = digest.lower()
            if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
                raise ValueError("materialized benchmark hash must be SHA-256 hex")
        return self


class ProtectedBenchmarkSource(StrictModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,79}$")
    kind: Literal["jsonl_gzip", "jsonl", "json", "parquet", "huggingface_dataset", "builtin_craftly"]
    revision: str = Field(min_length=7, max_length=128)
    license: str = Field(min_length=2, max_length=100)
    output_file: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*\.jsonl$")
    minimum_rows: int = Field(ge=1)
    maximum_bytes: int = Field(default=100_000_000, ge=1_000, le=2_000_000_000)
    url: str | None = None
    dataset_id: str | None = None
    split: str = "test"
    subset: str | None = None
    train_policy: Literal["eval_holdout"] = "eval_holdout"

    @model_validator(mode="after")
    def validate_source(self) -> "ProtectedBenchmarkSource":
        if ".." in Path(self.output_file).parts or Path(self.output_file).is_absolute():
            raise ValueError("benchmark output_file must stay inside the target directory")
        if self.kind in {"jsonl_gzip", "jsonl", "json", "parquet"}:
            if not self.url:
                raise ValueError(f"{self.name} requires url")
            parsed = urlparse(self.url)
            if parsed.scheme != "https" or parsed.username or parsed.password or not parsed.hostname:
                raise ValueError(f"{self.name} requires a credential-free HTTPS URL")
            if self.revision not in parsed.path:
                raise ValueError(f"{self.name} URL must contain its immutable revision")
        elif self.kind == "huggingface_dataset":
            if not self.dataset_id:
                raise ValueError(f"{self.name} requires dataset_id")
        elif self.url or self.dataset_id:
            raise ValueError("builtin benchmark cannot declare a remote source")
        return self


class ProtectedBenchmarkConfig(StrictModel):
    schema_version: Literal[1] = 1
    pack_id: str = Field(min_length=3, max_length=120)
    sources: list[ProtectedBenchmarkSource] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_sources(self) -> "ProtectedBenchmarkConfig":
        names = [source.name for source in self.sources]
        outputs = [source.output_file.lower() for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("protected benchmark names must be unique")
        if len(outputs) != len(set(outputs)):
            raise ValueError("protected benchmark output files must be unique")
        return self


class ProtectedBenchmarkArtifact(StrictModel):
    name: str
    path: str
    sha256: str
    rows: int = Field(ge=1)
    revision: str
    license: str
    train_policy: Literal["eval_holdout"] = "eval_holdout"

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError("artifact sha256 must be SHA-256 hex")
        return normalized


class ProtectedBenchmarkManifest(StrictModel):
    schema_version: Literal[1] = 1
    pack_id: str
    status: Literal["passed"]
    config_sha256: str
    artifacts: list[ProtectedBenchmarkArtifact]
    fingerprint_algorithm: Literal["minhash-v2-one-permutation-64"]
    fingerprint_index_path: str
    fingerprint_index_sha256: str
    created_at_unix: float = Field(default_factory=time.time)


PUBLIC_BENCHMARK_SOURCES = {
    "humaneval": {
        "url": (
            "https://raw.githubusercontent.com/openai/human-eval/"
            "6d43fb980f9fee3c892a914eda09951f772ad10d/data/HumanEval.jsonl.gz"
        ),
        "revision": "6d43fb980f9fee3c892a914eda09951f772ad10d",
        "license": "MIT",
        "minimum_rows": 164,
    },
    "mbpp": {
        "url": (
            "https://raw.githubusercontent.com/google-research/google-research/"
            "95e3a1da2d27cb9c8289f6fd3076cfed608c3c94/mbpp/mbpp.jsonl"
        ),
        "revision": "95e3a1da2d27cb9c8289f6fd3076cfed608c3c94",
        "license": "Apache-2.0",
        "minimum_rows": 374,
    },
}


REMOTE_BENCHMARK_DOMAINS = {
    "raw.githubusercontent.com",
    "huggingface.co",
    "cdn-lfs.huggingface.co",
    "cas-bridge.xethub.hf.co",
}
HUGGINGFACE_DATASETS = {"princeton-nlp/SWE-bench_Verified"}


def _allowed_remote_benchmark_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    host = hostname.lower()
    return host in REMOTE_BENCHMARK_DOMAINS or host.endswith(".aws.cdn.hf.co")


def _download_bytes(url: str, *, max_bytes: int = 20_000_000) -> bytes:
    """Download only pinned benchmark assets from the fixed public allowlist."""

    current = url
    for _ in range(5):
        parsed = urlparse(current)
        if (
            parsed.scheme != "https"
            or not _allowed_remote_benchmark_host(parsed.hostname)
            or parsed.username
            or parsed.password
        ):
            raise ValueError(f"benchmark URL is outside the HTTPS allowlist: {current}")
        with httpx.stream(
            "GET",
            current,
            follow_redirects=False,
            timeout=httpx.Timeout(60.0, connect=15.0),
            headers={"User-Agent": "CraftlyBenchmarkMaterializer/2.0"},
        ) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("benchmark redirect omitted Location")
                current = str(httpx.URL(current).join(location))
                continue
            response.raise_for_status()
            payload = bytearray()
            for chunk in response.iter_bytes():
                payload.extend(chunk)
                if len(payload) > max_bytes:
                    raise ValueError(f"benchmark download exceeded {max_bytes} bytes: {url}")
            return bytes(payload)
    raise ValueError(f"benchmark download exceeded redirect limit: {url}")


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def materialize_public_benchmark_pack(output_dir: str | Path) -> BenchmarkMaterializationReport:
    """Fetch public coding benchmarks into Craftly' local eval JSONL format."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    human_contract = PUBLIC_BENCHMARK_SOURCES["humaneval"]
    mbpp_contract = PUBLIC_BENCHMARK_SOURCES["mbpp"]
    human_payload = gzip.decompress(_download_bytes(str(human_contract["url"])))
    human_rows = [json.loads(line) for line in human_payload.decode("utf-8").splitlines() if line.strip()]
    mbpp_payload = _download_bytes(str(mbpp_contract["url"]))
    mbpp_rows = [json.loads(line) for line in mbpp_payload.decode("utf-8").splitlines() if line.strip()]
    files = {
        "humaneval": str(root / "humaneval.jsonl"),
        "mbpp": str(root / "mbpp.jsonl"),
    }
    rows = {
        "humaneval": _write_jsonl_rows(Path(files["humaneval"]), human_rows),
        "mbpp": _write_jsonl_rows(Path(files["mbpp"]), mbpp_rows),
    }
    minimums = {
        "humaneval": int(human_contract["minimum_rows"]),
        "mbpp": int(mbpp_contract["minimum_rows"]),
    }
    report = BenchmarkMaterializationReport(
        status="passed" if all(rows[name] >= minimums[name] for name in minimums) else "failed",
        output_dir=str(root),
        files=files,
        rows=rows,
        sources={
            "humaneval": str(human_contract["url"]),
            "mbpp": str(mbpp_contract["url"]),
        },
        revisions={
            "humaneval": str(human_contract["revision"]),
            "mbpp": str(mbpp_contract["revision"]),
        },
        licenses={
            "humaneval": str(human_contract["license"]),
            "mbpp": str(mbpp_contract["license"]),
        },
        sha256={name: _sha256_file(Path(path)) for name, path in files.items()},
        train_policy={"humaneval": "eval_holdout", "mbpp": "eval_holdout"},
    )
    (root / "benchmark_materialization_report.json").write_text(json.dumps(report.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
    return report


def _rows_from_payload(source: ProtectedBenchmarkSource, payload: bytes) -> list[dict[str, Any]]:
    if source.kind == "jsonl_gzip":
        payload = gzip.decompress(payload)
        return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if source.kind == "jsonl":
        return [json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
    if source.kind == "parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to materialize a protected Parquet benchmark") from exc
        table = parquet.read_table(io.BytesIO(payload))
        return [dict(row) for row in table.to_pylist()]
    parsed = json.loads(payload.decode("utf-8-sig"))
    if isinstance(parsed, list):
        return [dict(row) for row in parsed]
    if not isinstance(parsed, dict):
        raise ValueError(f"{source.name} JSON root must be an object or array")
    for candidate in ("data", "rows", "tasks", "test_cases", "prompts"):
        value = parsed.get(candidate)
        if isinstance(value, list):
            return [dict(row) for row in value]
    return [parsed]


def _load_huggingface_rows(source: ProtectedBenchmarkSource) -> list[dict[str, Any]]:
    if source.dataset_id not in HUGGINGFACE_DATASETS:
        raise ValueError(f"Hugging Face dataset is not allowlisted: {source.dataset_id}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to materialize the protected SWE-bench holdout") from exc
    dataset = load_dataset(
        source.dataset_id,
        source.subset,
        split=source.split,
        revision=source.revision,
        trust_remote_code=False,
    )
    return [dict(row) for row in dataset]


def load_protected_benchmark_config(path: str | Path) -> ProtectedBenchmarkConfig:
    return ProtectedBenchmarkConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def validate_protected_benchmark_manifest(
    config_path: str | Path,
    manifest_path: str | Path,
) -> ProtectedBenchmarkManifest:
    config_file = Path(config_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    config = load_protected_benchmark_config(config_file)
    manifest = ProtectedBenchmarkManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
    failures: list[str] = []
    if manifest.pack_id != config.pack_id:
        failures.append("pack_id mismatch")
    if manifest.config_sha256 != _sha256_file(config_file):
        failures.append("config hash mismatch")
    configured = {source.name: source for source in config.sources}
    if set(configured) != {artifact.name for artifact in manifest.artifacts}:
        failures.append("manifest artifact set does not match config")
    root = manifest_file.parent.resolve()
    for artifact in manifest.artifacts:
        source = configured.get(artifact.name)
        path = (root / artifact.path).resolve()
        if root not in path.parents:
            failures.append(f"{artifact.name}: artifact path escapes manifest directory")
            continue
        if source is None:
            continue
        if not path.is_file():
            failures.append(f"{artifact.name}: artifact missing")
            continue
        if _sha256_file(path) != artifact.sha256:
            failures.append(f"{artifact.name}: artifact hash mismatch")
        rows = _jsonl_count(str(path))
        if rows != artifact.rows or rows < source.minimum_rows:
            failures.append(f"{artifact.name}: row count mismatch or below minimum")
        if artifact.revision != source.revision or artifact.license.lower() != source.license.lower():
            failures.append(f"{artifact.name}: source contract mismatch")
    index_path = (root / manifest.fingerprint_index_path).resolve()
    if root not in index_path.parents or not index_path.is_file():
        failures.append("protected fingerprint index missing or outside manifest directory")
    elif _sha256_file(index_path) != manifest.fingerprint_index_sha256:
        failures.append("protected fingerprint index hash mismatch")
    if failures:
        raise ValueError("invalid protected benchmark manifest: " + "; ".join(failures))
    return manifest


def materialize_protected_benchmark_pack(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    downloader: Callable[..., bytes] = _download_bytes,
    huggingface_loader: Callable[[ProtectedBenchmarkSource], list[dict[str, Any]]] = _load_huggingface_rows,
) -> ProtectedBenchmarkManifest:
    """Materialize an immutable, eval-only benchmark pack and fingerprint DB."""

    config_file = Path(config_path).resolve()
    config = load_protected_benchmark_config(config_file)
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"protected benchmark output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    artifacts: list[ProtectedBenchmarkArtifact] = []
    holdouts: list[Path] = []
    for source in config.sources:
        if source.kind == "builtin_craftly":
            rows = [task.model_dump(mode="json") for task in built_in_security_tasks()]
        elif source.kind == "huggingface_dataset":
            rows = huggingface_loader(source)
        else:
            assert source.url is not None
            rows = _rows_from_payload(source, downloader(source.url, max_bytes=source.maximum_bytes))
        if len(rows) < source.minimum_rows:
            raise ValueError(
                f"{source.name} produced {len(rows)} rows; minimum is {source.minimum_rows}"
            )
        target = (root / source.output_file).resolve()
        if root not in target.parents:
            raise ValueError(f"benchmark output escapes target directory: {source.output_file}")
        _write_jsonl_rows(target, rows)
        holdouts.append(target)
        artifacts.append(
            ProtectedBenchmarkArtifact(
                name=source.name,
                path=target.relative_to(root).as_posix(),
                sha256=_sha256_file(target),
                rows=len(rows),
                revision=source.revision,
                license=source.license,
            )
        )
    fingerprint_path = root / "protected_fingerprints.sqlite3"
    build_protected_fingerprint_index(holdouts, fingerprint_path)
    manifest = ProtectedBenchmarkManifest(
        pack_id=config.pack_id,
        status="passed",
        config_sha256=_sha256_file(config_file),
        artifacts=artifacts,
        fingerprint_algorithm="minhash-v2-one-permutation-64",
        fingerprint_index_path=fingerprint_path.relative_to(root).as_posix(),
        fingerprint_index_sha256=_sha256_file(fingerprint_path),
    )
    manifest_path = root / "protected_benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_protected_benchmark_manifest(config_file, manifest_path)
    return manifest


def _spec(name: str, kind: str, path: str | None, *, required: bool) -> BenchmarkSuiteSpec | None:
    if not path:
        return None
    return BenchmarkSuiteSpec(name=name, kind=kind, path=path, required=required)  # type: ignore[arg-type]


def _jsonl_count(path: str | None) -> int:
    if not path or not Path(path).exists():
        return 0
    return sum(1 for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip())


def validate_production_benchmark_pack(config: BenchmarkPackConfig) -> list[str]:
    if not config.production:
        return []
    failures = []
    required = {
        "HumanEval": (config.human_eval_path, config.min_human_eval_tasks),
        "MBPP": (config.mbpp_path, config.min_mbpp_tasks),
        "SWE-Bench": (config.swe_bench_path, config.min_swe_bench_tasks),
        "CyberSecEval": (config.cyberseceval_path, config.min_cyberseceval_tasks),
    }
    for name, (path, minimum) in required.items():
        count = _jsonl_count(path)
        if count < minimum:
            failures.append(f"{name} requires at least {minimum} JSONL rows, found {count}")
    return failures


def run_benchmark_pack(config: BenchmarkPackConfig, *, output_dir: str | Path) -> BenchmarkPackReport:
    production_failures = validate_production_benchmark_pack(config)
    if production_failures:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        report = BenchmarkPackReport(
            status="failed",
            strict=config.strict,
            required_suites=[],
            optional_suites=[],
            suite_report={"status": "failed", "production_failures": production_failures},
            recommendations=production_failures,
        )
        (root / "benchmark_pack_report.json").write_text(json.dumps(report.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
        return report
    specs = [
        _spec("humaneval", "human_eval_style", config.human_eval_path, required=config.strict),
        _spec("mbpp", "mbpp_style", config.mbpp_path, required=config.strict),
        _spec("swe_bench", "swe_bench_style", config.swe_bench_path, required=config.strict),
        _spec("cyberseceval", "cyberseceval_style", config.cyberseceval_path, required=config.strict),
        _spec("custom_security", "custom_security", config.custom_security_path, required=False),
    ]
    active_specs = [item for item in specs if item is not None]
    if not active_specs:
        raise ValueError("at least one benchmark path is required")
    suite_report: BenchmarkSuitesReport = run_benchmark_suites(active_specs)
    root = Path(output_dir)
    suite_report.write(root)
    required = [item.name for item in active_specs if item.required]
    optional = [item.name for item in active_specs if not item.required]
    recommendations: list[str] = []
    missing_required = [item.name for item in suite_report.suites if item.status == "failed" and "missing" in item.reason]
    if missing_required:
        recommendations.append("Provide local benchmark JSONL files for required suites before claiming benchmark coverage.")
    if suite_report.aggregate_score < 0.75:
        recommendations.append("Investigate benchmark failures before promoting the checkpoint.")
    if "custom_security" not in optional:
        recommendations.append("Add Craftly-owned custom security regression suite for non-public holdout coverage.")
    report = BenchmarkPackReport(
        status=suite_report.status,
        strict=config.strict,
        required_suites=required,
        optional_suites=optional,
        suite_report=suite_report.model_dump(),
        recommendations=recommendations,
    )
    (root / "benchmark_pack_report.json").write_text(json.dumps(report.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Craftly HumanEval/MBPP/SWE/CyberSec benchmark pack.")
    parser.add_argument("--materialize-public", action="store_true", help="Download public HumanEval and MBPP JSONL files into --target-dir.")
    parser.add_argument("--materialize-protected", action="store_true", help="Build the pinned, eval-only protected benchmark pack.")
    parser.add_argument("--protected-config", default="config/protected_benchmarks.json")
    parser.add_argument("--validate-protected-manifest")
    parser.add_argument("--target-dir", default="data/eval")
    parser.add_argument("--human-eval")
    parser.add_argument("--mbpp")
    parser.add_argument("--swe-bench")
    parser.add_argument("--cyberseceval")
    parser.add_argument("--custom-security")
    parser.add_argument("--output-dir", default="artifacts/craftly/benchmark-pack")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--min-human-eval-tasks", type=int, default=164)
    parser.add_argument("--min-mbpp-tasks", type=int, default=374)
    parser.add_argument("--min-swe-bench-tasks", type=int, default=1)
    parser.add_argument("--min-cyberseceval-tasks", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.materialize_protected:
        manifest = materialize_protected_benchmark_pack(args.protected_config, args.target_dir)
        print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        return
    if args.validate_protected_manifest:
        manifest = validate_protected_benchmark_manifest(args.protected_config, args.validate_protected_manifest)
        print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        return
    if args.materialize_public:
        report = materialize_public_benchmark_pack(args.target_dir)
        print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
        if report.status != "passed":
            raise SystemExit(2)
        return
    report = run_benchmark_pack(
        BenchmarkPackConfig(
            human_eval_path=args.human_eval,
            mbpp_path=args.mbpp,
            swe_bench_path=args.swe_bench,
            cyberseceval_path=args.cyberseceval,
            custom_security_path=args.custom_security,
            strict=not args.non_strict,
            production=args.production,
            min_human_eval_tasks=args.min_human_eval_tasks,
            min_mbpp_tasks=args.min_mbpp_tasks,
            min_swe_bench_tasks=args.min_swe_bench_tasks,
            min_cyberseceval_tasks=args.min_cyberseceval_tasks,
        ),
        output_dir=args.output_dir,
    )
    print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
    if report.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

