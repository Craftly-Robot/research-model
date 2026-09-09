"""Native Craftly scratch checkpoint serving.

This is the production path for serving Craftly-owned scratch checkpoints before
vLLM/TensorRT conversion exists. It intentionally fails fast on missing assets
or incompatible tokenizer/model state instead of falling back to a mock model.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import Field

from src.craftly.identity.auth import AuthConfig, install_auth
from src.craftly.identity.quota import QuotaConfig, install_quota
from src.craftly.model_ops.checkpoint_compare import GenerationConfig, generate_text
from src.craftly.model_ops.foundation import CheckpointManifest, sha256_file
from src.craftly.model_ops.tokenizer_pipeline import load_tokenizer
from src.craftly.model_ops.torch_decoder import (
    CraftlyDecoderLM,
    ScratchDecoderConfig,
    load_trusted_checkpoint,
    require_torch,
    select_torch_device,
)
from src.craftly.observability import METRICS, install_observability
from src.craftly.shared.schemas import StrictModel

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=4_194_304)


class ChatCompletionRequest(StrictModel):
    model: str = "craftly-scratch"
    messages: list[ChatMessage] = Field(min_length=1, max_length=1_024)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    top_k: int = Field(default=20, ge=0, le=500)
    stream: bool = False


class NativeServingConfig(StrictModel):
    checkpoint_manifest: str
    tokenizer_path: str
    model_name: str = "craftly-scratch"
    device: str = "auto"
    require_tokenizer_hash_match: bool = True
    auth_enabled: bool = True
    quota_enabled: bool = True
    required_scope: str = Field(default="model:generate", min_length=1, max_length=128)
    max_prompt_characters: int = Field(default=131_072, ge=1_024, le=4_194_304)
    max_messages: int = Field(default=128, ge=1, le=1_024)
    max_queue_depth: int = Field(default=64, ge=0, le=100_000)
    max_concurrent_generations: int = Field(default=1, ge=1, le=128)
    queue_timeout_seconds: float = Field(default=10.0, ge=0.1, le=300.0)
    generation_timeout_seconds: float = Field(default=120.0, ge=1.0, le=3_600.0)
    reject_context_truncation: bool = True


class NativeServingState:
    def __init__(self, config: NativeServingConfig) -> None:
        require_torch()
        self.config = config
        self.manifest = CheckpointManifest.model_validate(
            json.loads(Path(config.checkpoint_manifest).read_text(encoding="utf-8-sig"))
        )
        self.checkpoint_path = Path(self.manifest.checkpoint_dir) / "model.pt"
        self.tokenizer_path = Path(config.tokenizer_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"checkpoint model file not found: {self.checkpoint_path}"
            )
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer file not found: {self.tokenizer_path}")
        self.device = select_torch_device(config.device)
        payload = load_trusted_checkpoint(
            self.checkpoint_path, map_location=self.device
        )
        self.payload = payload
        self.model_config = ScratchDecoderConfig.model_validate(payload["config"])
        self.tokenizer = load_tokenizer(self.tokenizer_path)
        vocab_size = int(self.tokenizer.get_vocab_size(with_added_tokens=True))
        if vocab_size > self.model_config.vocab_size:
            raise ValueError(
                f"tokenizer vocab {vocab_size} exceeds model vocab {self.model_config.vocab_size}"
            )
        saved_hash = str(payload.get("tokenizer_sha256") or "")
        actual_hash = sha256_file(self.tokenizer_path)
        if (
            config.require_tokenizer_hash_match
            and saved_hash
            and saved_hash != actual_hash
        ):
            raise ValueError("tokenizer hash does not match checkpoint metadata")
        self.model = CraftlyDecoderLM(self.model_config).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.started_at_unix = time.time()
        self._generation_slots = asyncio.Semaphore(config.max_concurrent_generations)
        self._state_lock = asyncio.Lock()
        self._queued = 0
        self._active = 0
        self._completed = 0
        self._failed = 0
        self._timed_out = 0
        self._background_releases: set[asyncio.Task[None]] = set()

    def health(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "ready",
            "model_name": self.config.model_name,
            "checkpoint_manifest": self.config.checkpoint_manifest,
            "checkpoint_manifest_sha256": sha256_file(
                Path(self.config.checkpoint_manifest).expanduser().resolve(strict=True)
            ),
            "checkpoint_step": self.manifest.step,
            "trained_tokens": self.manifest.trained_tokens,
            "tokenizer_path": str(self.tokenizer_path),
            "tokenizer_sha256": sha256_file(self.tokenizer_path),
            "device": str(self.device),
            "uptime_seconds": round(time.time() - self.started_at_unix, 3),
            "generation_capacity": {
                "active": self._active,
                "queued": self._queued,
                "completed": self._completed,
                "failed": self._failed,
                "timed_out": self._timed_out,
                "max_concurrent": self.config.max_concurrent_generations,
                "max_queue_depth": self.config.max_queue_depth,
            },
            "scratch_only": True,
        }
        if self.device.type == "cuda":
            assert torch is not None
            payload["cuda_memory"] = {
                "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                "max_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(self.device)
                ),
            }
        return payload

    def render_prompt(self, messages: list[ChatMessage]) -> str:
        rendered = []
        for message in messages:
            role = message.role.strip().lower()
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"unsupported message role: {message.role}")
            rendered.append(f"{role.title()}:\n{message.content.strip()}")
        rendered.append("Assistant:\n")
        return "\n\n".join(rendered)

    def generate(self, request: ChatCompletionRequest) -> tuple[str, int, float]:
        started = time.perf_counter()
        prompt = self.render_prompt(request.messages)
        if len(request.messages) > self.config.max_messages:
            raise HTTPException(
                status_code=413, detail="message count exceeds serving limit"
            )
        if len(prompt) > self.config.max_prompt_characters:
            raise HTTPException(
                status_code=413, detail="prompt exceeds serving character limit"
            )
        prompt_tokens = self.tokenizer.encode(prompt).ids
        if (
            self.config.reject_context_truncation
            and len(prompt_tokens) + request.max_tokens
            > self.model_config.max_sequence_length
        ):
            raise HTTPException(
                status_code=413,
                detail=(
                    "prompt plus requested completion exceeds the model context window; "
                    "reduce input or max_tokens"
                ),
            )
        text, token_count = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            device=self.device,
            config=GenerationConfig(
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
            ),
        )
        return text, token_count, (time.perf_counter() - started) * 1000

    async def generate_async(
        self, request: ChatCompletionRequest
    ) -> tuple[str, int, float]:
        async with self._state_lock:
            if self._queued >= self.config.max_queue_depth:
                METRICS.inc("craftly_serving_rejected_total", reason="queue_full")
                raise HTTPException(status_code=503, detail="generation queue is full")
            self._queued += 1
            METRICS.set_gauge("craftly_serving_queue_depth", float(self._queued))
        try:
            await asyncio.wait_for(
                self._generation_slots.acquire(),
                timeout=self.config.queue_timeout_seconds,
            )
        except TimeoutError as exc:
            async with self._state_lock:
                self._queued = max(0, self._queued - 1)
                self._failed += 1
                METRICS.set_gauge("craftly_serving_queue_depth", float(self._queued))
            METRICS.inc("craftly_serving_rejected_total", reason="queue_timeout")
            raise HTTPException(
                status_code=503, detail="generation queue wait timed out"
            ) from exc

        async with self._state_lock:
            self._queued = max(0, self._queued - 1)
            self._active += 1
            METRICS.set_gauge("craftly_serving_queue_depth", float(self._queued))
            METRICS.set_gauge("craftly_serving_active_generations", float(self._active))
        work = asyncio.create_task(asyncio.to_thread(self.generate, request))
        try:
            result = await asyncio.wait_for(
                asyncio.shield(work),
                timeout=self.config.generation_timeout_seconds,
            )
        except TimeoutError as exc:
            async with self._state_lock:
                self._timed_out += 1
            METRICS.inc("craftly_serving_generation_timeouts_total")
            release_task = asyncio.create_task(self._release_after_completion(work))
            background_releases = getattr(self, "_background_releases", None)
            if background_releases is None:
                background_releases = set()
                self._background_releases = background_releases
            background_releases.add(release_task)
            release_task.add_done_callback(background_releases.discard)
            raise HTTPException(status_code=504, detail="generation timed out") from exc
        except Exception:
            await self._release_slot(failed=True)
            raise
        await self._release_slot(failed=False)
        return result

    async def _release_after_completion(
        self, work: "asyncio.Task[tuple[str, int, float]]"
    ) -> None:
        with contextlib.suppress(Exception):
            await work
        await self._release_slot(failed=True)

    async def _release_slot(self, *, failed: bool) -> None:
        async with self._state_lock:
            self._active = max(0, self._active - 1)
            if failed:
                self._failed += 1
            else:
                self._completed += 1
            METRICS.set_gauge("craftly_serving_active_generations", float(self._active))
        self._generation_slots.release()


def create_app(config: NativeServingConfig | None = None) -> FastAPI:
    active_config = config or NativeServingConfig(
        checkpoint_manifest=os.environ.get("CRAFTLY_CHECKPOINT_MANIFEST", ""),
        tokenizer_path=os.environ.get("CRAFTLY_TOKENIZER_PATH", ""),
        model_name=os.environ.get("CRAFTLY_MODEL_NAME", "craftly-scratch"),
        device=os.environ.get("CRAFTLY_SERVING_DEVICE", "auto"),
        require_tokenizer_hash_match=os.environ.get(
            "CRAFTLY_REQUIRE_TOKENIZER_HASH_MATCH", "1"
        )
        == "1",
        auth_enabled=os.environ.get("CRAFTLY_AUTH_ENABLED", "1") == "1",
        quota_enabled=os.environ.get("CRAFTLY_QUOTA_ENABLED", "1") == "1",
        max_prompt_characters=int(
            os.environ.get("CRAFTLY_MAX_PROMPT_CHARACTERS", "131072")
        ),
        max_messages=int(os.environ.get("CRAFTLY_MAX_MESSAGES", "128")),
        max_queue_depth=int(os.environ.get("CRAFTLY_MAX_QUEUE_DEPTH", "64")),
        max_concurrent_generations=int(
            os.environ.get("CRAFTLY_MAX_CONCURRENT_GENERATIONS", "1")
        ),
        queue_timeout_seconds=float(
            os.environ.get("CRAFTLY_QUEUE_TIMEOUT_SECONDS", "10")
        ),
        generation_timeout_seconds=float(
            os.environ.get("CRAFTLY_GENERATION_TIMEOUT_SECONDS", "120")
        ),
    )
    if not active_config.checkpoint_manifest:
        raise ValueError("CRAFTLY_CHECKPOINT_MANIFEST is required")
    if not active_config.tokenizer_path:
        raise ValueError("CRAFTLY_TOKENIZER_PATH is required")
    state = NativeServingState(active_config)
    app = FastAPI(title="Craftly Native Scratch Serving", version="1.0.0")
    if active_config.quota_enabled:
        quota_config = QuotaConfig.from_env()
        if not quota_config.enabled or not quota_config.redis_url:
            raise RuntimeError(
                "native production serving requires CRAFTLY_QUOTA_ENABLED=1 and CRAFTLY_REDIS_URL"
            )
        install_quota(app, quota_config)
    if active_config.auth_enabled:
        auth_config = AuthConfig.from_env()
        if not auth_config.enabled or not auth_config.jwt_secret:
            raise RuntimeError(
                "native production serving requires CRAFTLY_AUTH_ENABLED=1 and CRAFTLY_JWT_SECRET"
            )
        install_auth(app, auth_config)
    install_observability(app)

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready() -> dict[str, Any]:
        return state.health()

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "data": [
                {
                    "id": active_config.model_name,
                    "object": "model",
                    "owned_by": "craftly",
                }
            ]
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return METRICS.render_prometheus()

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request) -> Any:
        if payload.model != active_config.model_name:
            raise HTTPException(
                status_code=404, detail=f"unknown model: {payload.model}"
            )
        if active_config.auth_enabled:
            scopes = set(getattr(request.state, "jwt_claims", {}).get("scopes", []))
            if (
                active_config.required_scope not in scopes
                and "training:admin" not in scopes
            ):
                raise HTTPException(
                    status_code=403, detail="missing model generation scope"
                )
        if payload.stream:
            return StreamingResponse(
                _stream_response(state, payload), media_type="text/event-stream"
            )
        text, token_count, latency_ms = await state.generate_async(payload)
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": active_config.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "length",
                }
            ],
            "usage": {"completion_tokens": token_count, "total_tokens": token_count},
            "craftly": {"latency_ms": round(latency_ms, 3), "scratch_only": True},
        }

    return app


async def _stream_response(
    state: NativeServingState, request: ChatCompletionRequest
) -> Any:
    text, token_count, latency_ms = await state.generate_async(request)
    chunk_id = f"chatcmpl-{uuid.uuid4()}"
    for part in text.split():
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": state.config.model_name,
            "choices": [
                {"index": 0, "delta": {"content": part + " "}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0)
    final = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": state.config.model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
        "usage": {"completion_tokens": token_count, "total_tokens": token_count},
        "craftly": {"latency_ms": round(latency_ms, 3), "scratch_only": True},
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a Craftly scratch checkpoint with a native OpenAI-compatible API."
    )
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-name", default="craftly-scratch")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument("--no-quota", action="store_true")
    parser.add_argument("--allow-tokenizer-hash-mismatch", action="store_true")
    parser.add_argument("--max-prompt-characters", type=int, default=131_072)
    parser.add_argument("--max-messages", type=int, default=128)
    parser.add_argument("--max-queue-depth", type=int, default=64)
    parser.add_argument("--max-concurrent-generations", type=int, default=1)
    parser.add_argument("--queue-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--generation-timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import uvicorn

    app = create_app(
        NativeServingConfig(
            checkpoint_manifest=args.checkpoint_manifest,
            tokenizer_path=args.tokenizer_path,
            model_name=args.model_name,
            device=args.device,
            require_tokenizer_hash_match=not args.allow_tokenizer_hash_mismatch,
            auth_enabled=not args.no_auth,
            quota_enabled=not args.no_quota,
            max_prompt_characters=args.max_prompt_characters,
            max_messages=args.max_messages,
            max_queue_depth=args.max_queue_depth,
            max_concurrent_generations=args.max_concurrent_generations,
            queue_timeout_seconds=args.queue_timeout_seconds,
            generation_timeout_seconds=args.generation_timeout_seconds,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
