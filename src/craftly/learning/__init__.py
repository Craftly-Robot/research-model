"""Lazy facade for the Craftly learning pipeline.

The CLI modules in this package are designed to run with ``python -m``.
Keeping this package initializer lazy prevents those modules from being
preloaded before runpy executes them.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ContaminationDetector": (
        "src.craftly.learning.benchmark_contamination_filter",
        "ContaminationDetector",
    ),
    "ContaminationReport": (
        "src.craftly.learning.benchmark_contamination_filter",
        "ContaminationReport",
    ),
    "CrawlConfig": ("src.craftly.learning.web_ingest", "CrawlConfig"),
    "DataEngine": ("src.craftly.learning.data_engine", "DataEngine"),
    "DataEngineConfig": ("src.craftly.learning.data_engine", "DataEngineConfig"),
    "DataEngineReport": ("src.craftly.learning.data_engine", "DataEngineReport"),
    "DataPipelineConfig": ("src.craftly.learning.data_pipeline", "DataPipelineConfig"),
    "DataPipelineReport": ("src.craftly.learning.data_pipeline", "DataPipelineReport"),
    "DatasetLedger": ("src.craftly.learning.versioning", "DatasetLedger"),
    "DatasetQualityGate": ("src.craftly.learning.quality", "DatasetQualityGate"),
    "DatasetVersionManifest": ("src.craftly.learning.versioning", "DatasetVersionManifest"),
    "ExtractedTask": ("src.craftly.learning.task_extraction", "ExtractedTask"),
    "FrontierStore": ("src.craftly.learning.data_engine", "FrontierStore"),
    "IngestReport": ("src.craftly.learning.web_ingest", "IngestReport"),
    "LocalObjectStore": ("src.craftly.learning.storage", "LocalObjectStore"),
    "ObjectStoreConfig": ("src.craftly.learning.storage", "ObjectStoreConfig"),
    "QualityGateConfig": ("src.craftly.learning.quality", "QualityGateConfig"),
    "QualityGateReport": ("src.craftly.learning.quality", "QualityGateReport"),
    "S3ObjectStore": ("src.craftly.learning.storage", "S3ObjectStore"),
    "SourceRegistry": ("src.craftly.learning.source_registry", "SourceRegistry"),
    "SourceRegistryReport": ("src.craftly.learning.source_registry", "SourceRegistryReport"),
    "SourceSpec": ("src.craftly.learning.web_ingest", "SourceSpec"),
    "TaskExtractionReport": ("src.craftly.learning.task_extraction", "TaskExtractionReport"),
    "WebCorpusIngestor": ("src.craftly.learning.web_ingest", "WebCorpusIngestor"),
    "extract_tasks": ("src.craftly.learning.task_extraction", "extract_tasks"),
    "run_data_pipeline": ("src.craftly.learning.data_pipeline", "run_data_pipeline"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(
        import_module(module_name), attr_name
    )  # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
    globals()[name] = value
    return value
