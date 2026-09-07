"""Tokenizer training and token sharding pipeline for Craftly scratch pretraining."""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import Field

from src.craftly.shared.schemas import StrictModel
from src.craftly.shared.integrity import sha256_file


SPECIAL_TOKENS = [
    "<|document_end|>",
    "<|thought_start|>",
    "<|thought_end|>",
    "<|call_graph_root|>",
    "<|patch_start|>",
    "<|patch_end|>",
    "<|compile_error|>",
    "<|exploit_success|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|heap_alloc|>",
    "<|heap_free|>",
    "<|stack_frame|>",
    "<|memory_address|>",
]


class TokenizerTrainConfig(StrictModel):
    vocab_size: int = Field(default=128_000, ge=1_000)
    min_frequency: int = Field(default=2, ge=1)
    special_tokens: list[str] = Field(default_factory=lambda: SPECIAL_TOKENS.copy())


class ShardBuildConfig(StrictModel):
    shard_token_count: int = Field(default=1_000_000, ge=128)
    sequence_length: int = Field(default=2048, ge=16)
    validation_fraction: float = Field(default=0.01, ge=0.0, le=0.5)
    seed: int = 1337
    document_boundary_token: str = "<|document_end|>"


class ShardManifest(StrictModel):
    schema_version: int = Field(default=1, ge=1, le=2)
    dataset_id: str
    tokenizer_path: str
    output_dir: str
    train_shards: list[str]
    val_shards: list[str]
    train_tokens: int
    val_tokens: int
    sequence_length: int
    shard_sha256: dict[str, str] = Field(default_factory=dict)
    tokenizer_sha256: str = ""
    dataset_manifest_path: str = ""
    dataset_manifest_sha256: str = ""
    source_sha256: dict[str, str] = Field(default_factory=dict)
    split_strategy: Literal["random_row", "pre_split_family_safe"] = "random_row"
    document_boundary_token: str = ""
    document_boundary_token_id: int | None = Field(default=None, ge=0)
    train_documents: int = Field(default=0, ge=0)
    val_documents: int = Field(default=0, ge=0)
    boundary_token_count: int = Field(default=0, ge=0)
    created_at_unix: float = Field(default_factory=time.time)


TOKENIZER_STRESS_TEXT = """
<|thought_start|> inspect call graph <|thought_end|>
<|call_graph_root|> auth.validate -> db.query -> cache.set
<|compile_error|> gcc error: undefined reference at 0x7ffd00ff
<|heap_alloc|> malloc calloc realloc free <|heap_free|>
0x00 0x01 0x02 0x03 0x7f 0x80 0xff 0xdeadbeef 0x7ffd00ff
def python_block():
  two_space_indent = True
    four_space_indent = True
        eight_space_indent = True
fn rust_block() { println!("hello"); }
int main(void) { char buf[16]; return 0; }
#!/usr/bin/env bash
set -euo pipefail
"""


class RealCorpusTokenizerConfig(StrictModel):
    input_paths: list[str]
    validation_input_paths: list[str] = Field(default_factory=list)
    output_dir: str
    dataset_id: str = "craftly-real-corpus"
    vocab_size: int = Field(default=128_000, ge=1_000)
    min_frequency: int = Field(default=2, ge=1)
    shard_token_count: int = Field(default=1_000_000, ge=128)
    sequence_length: int = Field(default=2048, ge=16)
    validation_fraction: float = Field(default=0.01, ge=0.0, le=0.5)
    include_stress_samples: bool = True
    require_exact_vocab_size: bool = True
    production_mode: bool = False
    dataset_manifest_path: str | None = None
    maximum_unknown_rate: float = Field(default=0.000001, ge=0.0, le=1.0)
    maximum_single_character_rate: float = Field(default=0.55, ge=0.0, le=1.0)
    maximum_whitespace_rate: float = Field(default=0.35, ge=0.0, le=1.0)
    maximum_punctuation_rate: float = Field(default=0.35, ge=0.0, le=1.0)


class TokenizerArtifactManifest(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["passed", "failed"]
    dataset_id: str
    dataset_manifest_path: str = ""
    dataset_manifest_sha256: str = ""
    tokenizer_path: str
    tokenizer_sha256: str
    shard_manifest_path: str
    shard_manifest_sha256: str
    source_sha256: dict[str, str]
    split_strategy: Literal["random_row", "pre_split_family_safe"]
    family_safe_split: bool
    vocab_size: int
    special_tokens: list[str]
    created_at_unix: float = Field(default_factory=time.time)


class TokenizerAuditReport(StrictModel):
    status: str
    tokenizer_path: str
    shard_manifest_path: str
    vocab_size_requested: int
    vocab_size_actual: int
    special_tokens_missing: list[str]
    audit_failures: list[str] = Field(default_factory=list)
    sample_token_counts: dict[str, int]
    source_rows: int
    source_chars: int
    shard_manifest: dict[str, Any]
    tokenizer_sha256: str = ""
    tokenizer_manifest_path: str = ""
    tokenizer_manifest_sha256: str = ""
    shard_manifest_sha256: str = ""
    dataset_manifest_path: str = ""
    dataset_manifest_sha256: str = ""
    source_sha256: dict[str, str] = Field(default_factory=dict)
    split_strategy: Literal["random_row", "pre_split_family_safe"] = "random_row"
    family_safe_split: bool = False
    token_statistics: dict[str, float | int] = Field(default_factory=dict)
    language_efficiency: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    efficiency_report_path: str = ""
    efficiency_report_sha256: str = ""
    created_at_unix: float = Field(default_factory=time.time)


def _write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _resolved_hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve(strict=True)
        if str(path) in result:
            continue
        result[str(path)] = sha256_file(path)
    return result


def iter_texts(paths: list[str | Path]) -> Iterable[str]:
    for path in paths:
        source = Path(path)
        if source.suffix == ".jsonl":
            with source.open("r", encoding="utf-8-sig", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        snippet = line[:240].replace("\n", "\\n")
                        raise ValueError(
                            f"invalid JSONL in {source} at line {line_number}: {exc.msg}; snippet={snippet!r}"
                        ) from exc
                    text = str(row.get("text") or row.get("content") or row.get("prompt") or "")
                    if text:
                        yield text
        else:
            yield source.read_text(encoding="utf-8", errors="replace")


def train_bpe_tokenizer(input_paths: list[str | Path], output_path: str | Path, config: TokenizerTrainConfig | None = None) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    active = config or TokenizerTrainConfig()
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=active.vocab_size,
        min_frequency=active.min_frequency,
        special_tokens=["<unk>", *active.special_tokens],
        show_progress=False,
    )
    tokenizer.train_from_iterator(iter_texts(input_paths), trainer=trainer)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(target))
    return target


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def write_uint32_tokens(path: Path, tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for token in tokens:
            handle.write(struct.pack("<I", int(token)))


def read_uint32_tokens(path: str | Path) -> list[int]:
    raw = Path(path).read_bytes()
    if len(raw) % 4 != 0:
        raise ValueError(f"token shard byte length is not divisible by 4: {path}")
    return [item[0] for item in struct.iter_unpack("<I", raw)]


def build_token_shards(
    *,
    input_paths: list[str | Path],
    tokenizer_path: str | Path,
    output_dir: str | Path,
    config: ShardBuildConfig | None = None,
    dataset_id: str = "craftly-corpus",
    validation_input_paths: list[str | Path] | None = None,
    dataset_manifest_path: str | Path | None = None,
) -> ShardManifest:
    active = config or ShardBuildConfig()
    rng = random.Random(active.seed)
    tokenizer = load_tokenizer(tokenizer_path)
    boundary_token_id = tokenizer.token_to_id(active.document_boundary_token)
    if boundary_token_id is None:
        raise ValueError(
            "tokenizer is missing the configured document boundary token: "
            f"{active.document_boundary_token}"
        )
    root = Path(output_dir)
    train_dir = root / "train"
    val_dir = root / "val"
    train_shards: list[str] = []
    val_shards: list[str] = []
    train_tokens = val_tokens = 0
    train_buffer: list[int] = []
    val_buffer: list[int] = []
    shard_sha256: dict[str, str] = {}
    train_documents = val_documents = 0

    def flush(buffer: list[int], split: str) -> None:
        nonlocal train_tokens, val_tokens
        if not buffer:
            return
        directory = train_dir if split == "train" else val_dir
        index = len(train_shards) if split == "train" else len(val_shards)
        path = directory / f"shard-{index:06d}.bin"
        write_uint32_tokens(path, buffer.copy())
        shard_sha256[str(path)] = sha256_file(path)
        if split == "train":
            train_shards.append(str(path))
            train_tokens += len(buffer)
        else:
            val_shards.append(str(path))
            val_tokens += len(buffer)
        buffer.clear()

    def encode_into(paths: list[str | Path], *, forced_split: str | None = None) -> None:
        nonlocal train_documents, val_documents
        for text in iter_texts(paths):
            token_ids = tokenizer.encode(text).ids
            if not token_ids:
                continue
            token_ids.append(boundary_token_id)
            split = forced_split or ("val" if rng.random() < active.validation_fraction else "train")
            buffer = val_buffer if split == "val" else train_buffer
            buffer.extend(token_ids)
            if split == "val":
                val_documents += 1
            else:
                train_documents += 1
            while len(buffer) >= active.shard_token_count:
                chunk = buffer[: active.shard_token_count]
                del buffer[: active.shard_token_count]
                flush(chunk, split)

    validation_paths = list(validation_input_paths or [])
    if validation_paths:
        encode_into(list(input_paths), forced_split="train")
        encode_into(validation_paths, forced_split="val")
    else:
        encode_into(list(input_paths))
    flush(train_buffer, "train")
    flush(val_buffer, "val")
    all_sources = [*input_paths, *validation_paths]
    resolved_dataset_manifest = (
        Path(dataset_manifest_path).expanduser().resolve(strict=True)
        if dataset_manifest_path
        else None
    )
    manifest = ShardManifest(
        schema_version=2,
        dataset_id=dataset_id,
        tokenizer_path=str(tokenizer_path),
        output_dir=str(root),
        train_shards=train_shards,
        val_shards=val_shards,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        sequence_length=active.sequence_length,
        shard_sha256=shard_sha256,
        tokenizer_sha256=sha256_file(tokenizer_path),
        dataset_manifest_path=str(resolved_dataset_manifest or ""),
        dataset_manifest_sha256=(
            sha256_file(resolved_dataset_manifest) if resolved_dataset_manifest else ""
        ),
        source_sha256=_resolved_hashes(all_sources),
        split_strategy="pre_split_family_safe" if validation_paths else "random_row",
        document_boundary_token=active.document_boundary_token,
        document_boundary_token_id=boundary_token_id,
        train_documents=train_documents,
        val_documents=val_documents,
        boundary_token_count=train_documents + val_documents,
    )
    _write_json_atomic(root / "manifest.json", manifest.model_dump(mode="json"))
    return manifest


def corpus_stats(paths: list[str | Path]) -> tuple[int, int]:
    rows = 0
    chars = 0
    for path in paths:
        source = Path(path)
        if source.suffix == ".jsonl":
            with source.open("r", encoding="utf-8-sig", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        snippet = line[:240].replace("\n", "\\n")
                        raise ValueError(
                            f"invalid JSONL in {source} at line {line_number}: {exc.msg}; snippet={snippet!r}"
                        ) from exc
                    text = str(row.get("text") or row.get("content") or row.get("prompt") or "")
                    if text:
                        rows += 1
                        chars += len(text)
        else:
            text = source.read_text(encoding="utf-8", errors="replace")
            rows += 1
            chars += len(text)
    return rows, chars


def write_tokenizer_stress_file(root: str | Path) -> Path:
    target = Path(root) / "tokenizer_stress_samples.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"text": TOKENIZER_STRESS_TEXT}, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _load_production_dataset_binding(
    config: RealCorpusTokenizerConfig,
) -> tuple[str, dict[str, Any]]:
    if not config.dataset_manifest_path:
        if config.production_mode:
            raise ValueError("production tokenizer training requires dataset_manifest_path")
        return "", {}
    manifest_path = Path(config.dataset_manifest_path).expanduser().resolve(strict=True)
    if manifest_path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("dataset manifest exceeds 8 MiB")
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must be a JSON object")
    if config.production_mode:
        if payload.get("status") != "promoted" or payload.get("dev_smoke") is not False:
            raise ValueError("production tokenizer requires a non-smoke promoted dataset")
        if int((payload.get("metrics") or {}).get("promoted_records", 0)) < 100_000:
            raise ValueError("production tokenizer requires at least 100,000 promoted records")
        artifacts = payload.get("artifacts")
        artifact_sha256 = payload.get("artifact_sha256")
        if not isinstance(artifacts, dict) or not isinstance(artifact_sha256, dict):
            raise ValueError("dataset manifest artifact bindings are missing")
        required_artifacts = {"train", "val", "split_manifest", "promotion_decision"}
        missing = sorted(required_artifacts - artifacts.keys())
        if missing:
            raise ValueError(f"dataset manifest is missing required artifacts: {missing}")
        for name in required_artifacts:
            artifact = Path(str(artifacts[name])).expanduser().resolve(strict=True)
            expected = str(artifact_sha256.get(name) or "")
            if len(expected) != 64 or sha256_file(artifact) != expected:
                raise ValueError(f"dataset artifact integrity failed: {name}")
        train_paths = {str(Path(path).expanduser().resolve(strict=True)) for path in config.input_paths}
        val_paths = {
            str(Path(path).expanduser().resolve(strict=True))
            for path in config.validation_input_paths
        }
        if train_paths != {str(Path(str(artifacts["train"])).expanduser().resolve(strict=True))}:
            raise ValueError("tokenizer train input is not the promoted dataset train split")
        if val_paths != {str(Path(str(artifacts["val"])).expanduser().resolve(strict=True))}:
            raise ValueError("tokenizer validation input is not the promoted dataset validation split")
        split = (payload.get("reports") or {}).get("split_manifest") or {}
        if int(split.get("cross_split_group_collisions", -1)) != 0:
            raise ValueError("dataset split contains cross-family collisions")
        if not payload.get("advancement_decision_sha256"):
            raise ValueError("dataset manifest is not bound to the 5k advancement decision")
    return sha256_file(manifest_path), payload


def _token_audit_metrics(tokenizer: Any, paths: list[str | Path]) -> dict[str, float | int]:
    unknown_id = tokenizer.token_to_id("<unk>")
    total = unknown = single = whitespace = punctuation = 0
    sampled_chars = 0
    sampled_bytes = 0
    maximum_chars = 5_000_000
    for text in iter_texts(paths):
        if sampled_chars >= maximum_chars:
            break
        sample = text[: maximum_chars - sampled_chars]
        sampled_chars += len(sample)
        sampled_bytes += len(sample.encode("utf-8"))
        ids = tokenizer.encode(sample).ids
        for token_id in ids:
            decoded = tokenizer.decode([token_id])
            total += 1
            unknown += int(unknown_id is not None and token_id == unknown_id)
            single += int(len(decoded) == 1)
            whitespace += int(bool(decoded) and not decoded.strip())
            punctuation += int(
                bool(decoded)
                and all(not character.isalnum() and not character.isspace() for character in decoded)
            )
    denominator = max(1, total)
    return {
        "sampled_characters": sampled_chars,
        "sampled_bytes": sampled_bytes,
        "sampled_tokens": total,
        "unknown_tokens": unknown,
        "unknown_rate": unknown / denominator,
        "single_character_rate": single / denominator,
        "whitespace_token_rate": whitespace / denominator,
        "punctuation_token_rate": punctuation / denominator,
        "tokens_per_character": total / max(1, sampled_chars),
        "tokens_per_byte": total / max(1, sampled_bytes),
    }


def _language_efficiency(tokenizer: Any) -> dict[str, dict[str, float | int]]:
    samples = {
        "python": "def validate(value: str) -> str:\n    return value.strip()\n",
        "c_cpp": "int main(void) { char buf[16] = {0}; return (int)buf[0]; }\n",
        "rust": "fn checked_add(a: u64, b: u64) -> Option<u64> { a.checked_add(b) }\n",
        "go": "func Validate(value string) string { return strings.TrimSpace(value) }\n",
        "javascript": "export const validate = (value) => String(value).trim();\n",
        "bash": "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' \"$1\"\n",
        "patch": "@@ -1,2 +1,3 @@\n-old_call(user_input)\n+safe_call(validate(user_input))\n",
        "compiler_log": "<|compile_error|> error: undefined reference at 0x7ffd00ff\n",
        "hex_dump": "0x00 0x01 0x7f 0x80 0xff 0xdeadbeef 0x7ffd00ff\n",
        "indent_2": "if ready:\n  run()\n",
        "indent_4": "if ready:\n    run()\n",
        "indent_8": "if ready:\n        run()\n",
    }
    report: dict[str, dict[str, float | int]] = {}
    for name, sample in samples.items():
        count = len(tokenizer.encode(sample).ids)
        report[name] = {
            "characters": len(sample),
            "tokens": count,
            "tokens_per_character": count / max(1, len(sample)),
        }
    return report


def train_real_corpus_tokenizer(config: RealCorpusTokenizerConfig) -> TokenizerAuditReport:
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if config.production_mode and config.include_stress_samples:
        raise ValueError("production tokenizer corpus cannot include synthetic stress samples")
    dataset_manifest_sha256, _ = _load_production_dataset_binding(config)
    tokenizer_path = root / "tokenizer" / "tokenizer.json"
    shards_dir = root / "shards"
    input_paths: list[str | Path] = [Path(path) for path in config.input_paths]
    validation_paths: list[str | Path] = [Path(path) for path in config.validation_input_paths]
    if config.include_stress_samples:
        input_paths.append(write_tokenizer_stress_file(root))

    trained = train_bpe_tokenizer(
        input_paths,
        tokenizer_path,
        TokenizerTrainConfig(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            special_tokens=SPECIAL_TOKENS,
        ),
    )
    manifest = build_token_shards(
        input_paths=input_paths,
        tokenizer_path=trained,
        output_dir=shards_dir,
        config=ShardBuildConfig(
            shard_token_count=config.shard_token_count,
            sequence_length=config.sequence_length,
            validation_fraction=config.validation_fraction,
        ),
        dataset_id=config.dataset_id,
        validation_input_paths=validation_paths,
        dataset_manifest_path=config.dataset_manifest_path,
    )
    tokenizer = load_tokenizer(trained)
    vocab = tokenizer.get_vocab()
    actual_vocab_size = tokenizer.get_vocab_size()
    missing_special_tokens = [token for token in SPECIAL_TOKENS if token not in vocab]
    audit_failures = [f"missing special token: {token}" for token in missing_special_tokens]
    if config.require_exact_vocab_size and actual_vocab_size != config.vocab_size:
        audit_failures.append(
            f"vocabulary size mismatch: requested {config.vocab_size}, trained {actual_vocab_size}"
        )
    token_statistics = _token_audit_metrics(tokenizer, [*input_paths, *validation_paths])
    language_efficiency = _language_efficiency(tokenizer)
    if config.production_mode:
        thresholds = {
            "unknown_rate": config.maximum_unknown_rate,
            "single_character_rate": config.maximum_single_character_rate,
            "whitespace_token_rate": config.maximum_whitespace_rate,
            "punctuation_token_rate": config.maximum_punctuation_rate,
        }
        for metric, maximum in thresholds.items():
            if float(token_statistics[metric]) > maximum:
                audit_failures.append(
                    f"{metric} exceeded threshold: {token_statistics[metric]:.6f}>{maximum:.6f}"
                )
        if not validation_paths:
            audit_failures.append("production tokenizer requires a family-safe validation split")
    source_rows, source_chars = corpus_stats(config.input_paths)
    shard_manifest_path = shards_dir / "manifest.json"
    efficiency_path = _write_json_atomic(
        root / "token_efficiency_report.json",
        {
            "schema_version": 1,
            "dataset_id": config.dataset_id,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "token_statistics": token_statistics,
            "language_efficiency": language_efficiency,
            "thresholds": {
                "maximum_unknown_rate": config.maximum_unknown_rate,
                "maximum_single_character_rate": config.maximum_single_character_rate,
                "maximum_whitespace_rate": config.maximum_whitespace_rate,
                "maximum_punctuation_rate": config.maximum_punctuation_rate,
            },
        },
    )
    artifact_manifest = TokenizerArtifactManifest(
        status="passed" if not audit_failures else "failed",
        dataset_id=config.dataset_id,
        dataset_manifest_path=str(Path(config.dataset_manifest_path).resolve()) if config.dataset_manifest_path else "",
        dataset_manifest_sha256=dataset_manifest_sha256,
        tokenizer_path=str(trained.resolve()),
        tokenizer_sha256=sha256_file(trained),
        shard_manifest_path=str(shard_manifest_path.resolve()),
        shard_manifest_sha256=sha256_file(shard_manifest_path),
        source_sha256=manifest.source_sha256,
        split_strategy=manifest.split_strategy,
        family_safe_split=manifest.split_strategy == "pre_split_family_safe",
        vocab_size=actual_vocab_size,
        special_tokens=["<unk>", *SPECIAL_TOKENS],
    )
    tokenizer_manifest_path = _write_json_atomic(
        root / "tokenizer_manifest.json", artifact_manifest.model_dump(mode="json")
    )
    report = TokenizerAuditReport(
        status="passed" if not audit_failures else "failed",
        tokenizer_path=str(trained),
        shard_manifest_path=str(shard_manifest_path),
        vocab_size_requested=config.vocab_size,
        vocab_size_actual=actual_vocab_size,
        special_tokens_missing=missing_special_tokens,
        audit_failures=audit_failures,
        sample_token_counts={name: int(row["tokens"]) for name, row in language_efficiency.items()},
        source_rows=source_rows,
        source_chars=source_chars,
        shard_manifest=manifest.model_dump(),
        tokenizer_sha256=sha256_file(trained),
        tokenizer_manifest_path=str(tokenizer_manifest_path),
        tokenizer_manifest_sha256=sha256_file(tokenizer_manifest_path),
        shard_manifest_sha256=sha256_file(shard_manifest_path),
        dataset_manifest_path=str(Path(config.dataset_manifest_path).resolve()) if config.dataset_manifest_path else "",
        dataset_manifest_sha256=dataset_manifest_sha256,
        source_sha256=manifest.source_sha256,
        split_strategy=manifest.split_strategy,
        family_safe_split=manifest.split_strategy == "pre_split_family_safe",
        token_statistics=token_statistics,
        language_efficiency=language_efficiency,
        efficiency_report_path=str(efficiency_path),
        efficiency_report_sha256=sha256_file(efficiency_path),
    )
    _write_json_atomic(root / "tokenizer_audit_report.json", report.model_dump(mode="json"))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Craftly tokenizer and build token shards.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--tokenizer-out")
    parser.add_argument("--shards-out")
    parser.add_argument("--output-dir", help="Use with --real-corpus-audit to write tokenizer, shards, and audit report.")
    parser.add_argument("--dataset-id", default="craftly-corpus")
    parser.add_argument("--vocab-size", type=int, default=128_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--shard-token-count", type=int, default=1_000_000)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--validation-input", nargs="+", default=[])
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--production-mode", action="store_true")
    parser.add_argument("--real-corpus-audit", action="store_true")
    parser.add_argument("--no-stress-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.real_corpus_audit:
        if not args.output_dir:
            raise SystemExit("--output-dir is required with --real-corpus-audit")
        report = train_real_corpus_tokenizer(
            RealCorpusTokenizerConfig(
                input_paths=[str(path) for path in args.input],
                validation_input_paths=[str(path) for path in args.validation_input],
                output_dir=args.output_dir,
                dataset_id=args.dataset_id,
                vocab_size=args.vocab_size,
                min_frequency=args.min_frequency,
                shard_token_count=args.shard_token_count,
                sequence_length=args.sequence_length,
                validation_fraction=args.validation_fraction,
                include_stress_samples=not args.no_stress_samples,
                production_mode=args.production_mode,
                dataset_manifest_path=args.dataset_manifest,
            )
        )
        print(json.dumps(report.model_dump(), indent=2, sort_keys=True))
        if report.status != "passed":
            raise SystemExit(2)
        return
    if not args.tokenizer_out or not args.shards_out:
        raise SystemExit("--tokenizer-out and --shards-out are required unless --real-corpus-audit is used")
    tokenizer_path = train_bpe_tokenizer(
        args.input,
        args.tokenizer_out,
        TokenizerTrainConfig(vocab_size=args.vocab_size, min_frequency=args.min_frequency),
    )
    manifest = build_token_shards(
        input_paths=args.input,
        tokenizer_path=tokenizer_path,
        output_dir=args.shards_out,
        config=ShardBuildConfig(
            shard_token_count=args.shard_token_count,
            sequence_length=args.sequence_length,
            validation_fraction=args.validation_fraction,
        ),
        dataset_id=args.dataset_id,
    )
    print(json.dumps(manifest.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

