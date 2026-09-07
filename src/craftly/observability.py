"""Production observability primitives for Craftly."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts_unix": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ["method", "path", "status_code", "duration_ms", "user_id", "request_id"]:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger("craftly")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.setLevel(level)


@dataclass
class HistogramAggregate:
    count: int = 0
    total: float = 0.0
    maximum: float = float("-inf")

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)


@dataclass
class MetricsRegistry:
    counters: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    gauges: dict[str, float] = field(default_factory=dict)
    histograms: dict[str, HistogramAggregate] = field(default_factory=lambda: defaultdict(HistogramAggregate))
    lock: RLock = field(default_factory=RLock)

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = self._key(name, labels)
        with self.lock:
            self.counters[key] += value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = self._key(name, labels)
        with self.lock:
            self.histograms[key].observe(value)

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        key = self._key(name, labels)
        with self.lock:
            self.gauges[key] = value

    def inc_gauge(self, name: str, value: float = 1.0, **labels: str) -> float:
        key = self._key(name, labels)
        with self.lock:
            updated = max(0.0, self.gauges.get(key, 0.0) + value)
            self.gauges[key] = updated
            return updated

    def render_prometheus(self) -> str:
        lines = [
            "# HELP craftly_http_requests_total Total HTTP requests.",
            "# TYPE craftly_http_requests_total counter",
        ]
        with self.lock:
            for key, value in sorted(self.counters.items()):
                lines.append(f"{key} {value}")
            lines.extend(
                [
                    "# HELP craftly_http_requests_in_progress Requests currently being processed.",
                    "# TYPE craftly_http_requests_in_progress gauge",
                ]
            )
            for key, value in sorted(self.gauges.items()):
                lines.append(f"{key} {value}")
            lines.extend(
                [
                    "# HELP craftly_http_request_duration_ms HTTP request duration in milliseconds.",
                    "# TYPE craftly_http_request_duration_ms summary",
                ]
            )
            for key, aggregate in sorted(self.histograms.items()):
                if not aggregate.count:
                    continue
                lines.append(f'{self._with_labels(key, {"quantile": "avg"})} {aggregate.total / aggregate.count:.6f}')
                lines.append(f'{self._with_labels(key, {"quantile": "max"})} {aggregate.maximum:.6f}')
                lines.append(f"{self._count_key(key)} {aggregate.count}")
        return "\n".join(lines) + "\n"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    key: {"count": value.count, "max": value.maximum if value.count else 0.0}
                    for key, value in self.histograms.items()
                },
            }

    def _key(self, name: str, labels: dict[str, str]) -> str:
        if re.fullmatch(r"[a-zA-Z_:][a-zA-Z0-9_:]*", name) is None:
            raise ValueError(f"invalid Prometheus metric name: {name!r}")
        if not labels:
            return name
        label_text = ",".join(
            f'{key}="{self._escape_label(value)}"'
            for key, value in sorted(labels.items())
        )
        return f"{name}{{{label_text}}}"

    def _with_labels(self, metric_key: str, labels: dict[str, str]) -> str:
        extra = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
        if "{" not in metric_key:
            return f"{metric_key}{{{extra}}}"
        return metric_key[:-1] + f",{extra}" + "}"

    def _count_key(self, metric_key: str) -> str:
        if "{" not in metric_key:
            return f"{metric_key}_count"
        name, labels = metric_key.split("{", 1)
        return f"{name}_count{{{labels}"

    @staticmethod
    def _escape_label(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


METRICS = MetricsRegistry()
LOGGER = logging.getLogger("craftly.gateway")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        started = time.perf_counter()
        status_code = 500
        supplied_request_id = request.headers.get("x-request-id", "")
        request_id = (
            supplied_request_id
            if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", supplied_request_id)
            else str(uuid.uuid4())
        )
        request.state.request_id = request_id
        route_label = request.url.path
        METRICS.inc_gauge("craftly_http_requests_in_progress", 1.0)
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            route_object = request.scope.get("route")
            route_label = str(getattr(route_object, "path", route_label))
            method = request.method
            METRICS.inc_gauge("craftly_http_requests_in_progress", -1.0)
            METRICS.inc(
                "craftly_http_requests_total",
                method=method,
                path=route_label,
                status=str(status_code),
            )
            METRICS.observe(
                "craftly_http_request_duration_ms",
                duration_ms,
                method=method,
                path=route_label,
            )
            if status_code >= 500:
                METRICS.inc(
                    "craftly_http_server_errors_total",
                    method=method,
                    path=route_label,
                )
            LOGGER.info(
                "http_request",
                extra={
                    "method": method,
                    "path": route_label,
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 3),
                    "user_id": getattr(request.state, "user_id", ""),
                    "request_id": request_id,
                },
            )


def install_observability(app: FastAPI) -> None:
    configure_logging()
    app.add_middleware(ObservabilityMiddleware)
    install_tracing(app)


def install_tracing(app: FastAPI) -> None:
    endpoint = os.environ.get("CRAFTLY_OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logging.getLogger("craftly.gateway").warning("opentelemetry_not_installed")
        return
    provider = TracerProvider(resource=Resource.create({"service.name": os.environ.get("CRAFTLY_SERVICE_NAME", "craftly-api")}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)

