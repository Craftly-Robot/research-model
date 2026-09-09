"""Continuous regenerative quota enforcement."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from threading import RLock
from typing import Awaitable, Callable, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

QUOTA_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local capacity = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local tokens = tonumber(redis.call('HGET', key, 'tokens_balance') or capacity)
local last = tonumber(redis.call('HGET', key, 'last_updated_timestamp') or now)
local delta = math.max(0, now - last)
tokens = math.min(capacity, tokens + delta * rate)
local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end
redis.call('HSET', key, 'tokens_balance', tokens, 'last_updated_timestamp', now)
redis.call('EXPIRE', key, 2592000)
return {allowed, tokens}
"""

LOGGER = logging.getLogger("craftly.quota")


@dataclass(frozen=True)
class QuotaConfig:
    enabled: bool = False
    redis_url: str | None = None
    replenish_rate_per_second: float = 0.05
    capacity: float = 100.0
    default_cost: float = 1.0

    @classmethod
    def from_env(cls) -> "QuotaConfig":
        return cls(
            enabled=os.environ.get("CRAFTLY_QUOTA_ENABLED", "0") == "1",
            redis_url=os.environ.get("CRAFTLY_REDIS_URL"),
            replenish_rate_per_second=float(
                os.environ.get("CRAFTLY_QUOTA_RATE_PER_SECOND", "0.05")
            ),
            capacity=float(os.environ.get("CRAFTLY_QUOTA_CAPACITY", "100")),
            default_cost=float(os.environ.get("CRAFTLY_QUOTA_DEFAULT_COST", "1")),
        )


class LocalQuotaStore:
    def __init__(self) -> None:
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = RLock()

    def consume(
        self, subject: str, *, now: float, rate: float, capacity: float, cost: float
    ) -> tuple[bool, float]:
        with self._lock:
            tokens, last = self._state.get(subject, (capacity, now))
            tokens = min(capacity, tokens + max(0.0, now - last) * rate)
            allowed = tokens >= cost
            if allowed:
                tokens -= cost
            self._state[subject] = (tokens, now)
            return allowed, tokens


LOCAL_QUOTA = LocalQuotaStore()


class QuotaStore(Protocol):
    async def consume(
        self, subject: str, *, now: float, rate: float, capacity: float, cost: float
    ) -> tuple[bool, float]: ...


class AsyncLocalQuotaStore:
    async def consume(
        self, subject: str, *, now: float, rate: float, capacity: float, cost: float
    ) -> tuple[bool, float]:
        return LOCAL_QUOTA.consume(
            subject, now=now, rate=rate, capacity=capacity, cost=cost
        )


class RedisQuotaStore:
    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._client: object | None = None

    async def client(self) -> object:
        if self._client is None:
            from redis.asyncio import Redis

            self._client = Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    async def consume(
        self, subject: str, *, now: float, rate: float, capacity: float, cost: float
    ) -> tuple[bool, float]:
        client = await self.client()
        key = f"craftly:quota:{subject}"
        result = await client.eval(QUOTA_LUA, 1, key, now, rate, capacity, cost)
        allowed = bool(int(result[0]))
        remaining = float(result[1])
        return allowed, remaining


class QuotaMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, config: QuotaConfig) -> None:
        super().__init__(app)
        self.config = config
        self.store: QuotaStore = (
            RedisQuotaStore(config.redis_url)
            if config.redis_url
            else AsyncLocalQuotaStore()
        )

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if (
            not self.config.enabled
            or not request.url.path.startswith("/v1")
            or request.url.path.startswith("/v1/auth/")
        ):
            return await call_next(request)
        subject = str(getattr(request.state, "user_id", "anonymous"))
        cost = request_cost(request)
        try:
            allowed, remaining = await self.store.consume(
                subject,
                now=time.time(),
                rate=self.config.replenish_rate_per_second,
                capacity=self.config.capacity,
                cost=cost,
            )
        except Exception:
            LOGGER.exception(
                "quota_backend_unavailable",
                extra={"subject": subject, "path": request.url.path},
            )
            return JSONResponse(
                status_code=503, content={"error": "quota_backend_unavailable"}
            )
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "quota_exceeded", "remaining": remaining},
                headers={"X-Quota-Remaining": f"{remaining:.3f}"},
            )
        response = await call_next(request)
        response.headers["X-Quota-Remaining"] = f"{remaining:.3f}"
        response.headers["X-Quota-Cost"] = f"{cost:.3f}"
        return response


def request_cost(request: Request) -> float:
    path = request.url.path
    if "/agent/" in path:
        return 5.0
    if "/verifier/" in path or "/tools/" in path:
        return 3.0
    if "/context/" in path or "/index" in path:
        return 2.0
    return 1.0


def install_quota(app: FastAPI, config: QuotaConfig | None = None) -> QuotaConfig:
    active = config or QuotaConfig.from_env()
    app.add_middleware(QuotaMiddleware, config=active)
    return active
