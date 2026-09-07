"""Native Craftly JWT auth middleware.

Quota remains a gateway policy hook, but auth is production-shaped and
environment controlled.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware


@dataclass(frozen=True)
class AuthConfig:
    enabled: bool = False
    quota_enabled: bool = False
    jwt_secret: str | None = None
    allow_token_issue: bool = False
    token_issue_key: str | None = None
    issuer: str = "craftly-local"
    audience: str = "craftly-api"
    leeway_seconds: int = 30
    protected_prefixes: tuple[str, ...] = ("/v1",)
    exempt_paths: tuple[str, ...] = (
        "/",
        "/metrics",
        "/openapi.json",
        "/docs",
        "/redoc",
        "/health/live",
        "/health/ready",
        "/v1/auth/status",
        "/v1/auth/token",
        "/v1/training/token/exchange",
        "/v1/training/token/refresh",
    )

    @classmethod
    def from_env(cls) -> "AuthConfig":
        return cls(
            enabled=os.environ.get("CRAFTLY_AUTH_ENABLED", "0") == "1",
            quota_enabled=os.environ.get("CRAFTLY_QUOTA_ENABLED", "0") == "1",
            jwt_secret=os.environ.get("CRAFTLY_JWT_SECRET"),
            allow_token_issue=os.environ.get("CRAFTLY_ALLOW_TOKEN_ISSUE", "0") == "1",
            token_issue_key=os.environ.get("CRAFTLY_TOKEN_ISSUE_KEY"),
            issuer=os.environ.get("CRAFTLY_JWT_ISSUER", "craftly-local"),
            audience=os.environ.get("CRAFTLY_JWT_AUDIENCE", "craftly-api"),
        )


class AuthError(Exception):
    def __init__(self, code: str, detail: str, status_code: int = 401) -> None:
        self.code = code
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


def b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode((text + ("=" * (-len(text) % 4))).encode("ascii"))


def json_b64(payload: dict[str, Any]) -> str:
    return b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def sign_hs256(message: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), message.encode("ascii"), hashlib.sha256).digest()
    return b64url_encode(digest)


def create_jwt(
    *,
    subject: str,
    secret: str,
    issuer: str,
    audience: str,
    scopes: list[str] | None = None,
    ttl_seconds: int = 3600,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    issued_at = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "iss": issuer,
        "aud": audience,
        "iat": issued_at,
        "nbf": issued_at,
        "exp": issued_at + ttl_seconds,
        "scopes": scopes or ["api"],
    }
    reserved = {"sub", "iss", "aud", "iat", "nbf", "exp", "scopes"}
    for key, value in (extra_claims or {}).items():
        if key in reserved:
            raise ValueError(f"cannot override reserved JWT claim: {key}")
        payload[key] = value
    signing_input = f"{json_b64(header)}.{json_b64(payload)}"
    return f"{signing_input}.{sign_hs256(signing_input, secret)}"


def verify_jwt(token: str, *, config: AuthConfig) -> dict[str, Any]:
    if not config.jwt_secret:
        raise AuthError("auth_secret_missing", "JWT secret is not configured.", 503)
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError("invalid_token", "JWT must have three segments.")
    header_raw, payload_raw, signature = parts
    signing_input = f"{header_raw}.{payload_raw}"
    expected = sign_hs256(signing_input, config.jwt_secret)
    if not hmac.compare_digest(signature, expected):
        raise AuthError("invalid_signature", "JWT signature verification failed.")
    try:
        header = json.loads(b64url_decode(header_raw))
        payload = json.loads(b64url_decode(payload_raw))
    except (json.JSONDecodeError, ValueError) as exc:
        raise AuthError("invalid_payload", str(exc)) from exc
    now = int(time.time())
    if header.get("alg") != "HS256":
        raise AuthError("unsupported_alg", "Only HS256 JWTs are accepted.")
    if payload.get("iss") != config.issuer or payload.get("aud") != config.audience:
        raise AuthError("invalid_claims", "JWT issuer or audience mismatch.")
    if int(payload.get("nbf", 0)) > now + config.leeway_seconds:
        raise AuthError("not_yet_valid", "JWT is not active yet.")
    if int(payload.get("exp", 0)) < now - config.leeway_seconds:
        raise AuthError("expired_token", "JWT has expired.")
    if not payload.get("sub"):
        raise AuthError("missing_subject", "JWT subject is required.")
    return payload


def is_exempt(path: str, config: AuthConfig) -> bool:
    return path in config.exempt_paths or path.startswith("/static/")


def is_protected(path: str, config: AuthConfig) -> bool:
    return any(path.startswith(prefix) for prefix in config.protected_prefixes)


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, config: AuthConfig) -> None:
        super().__init__(app)
        self.config = config

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        path = request.url.path
        if is_exempt(path, self.config) or not is_protected(path, self.config):
            response = await call_next(request)
            response.headers["X-Auth-Mode"] = "exempt"
            return response
        if not self.config.enabled:
            response = await call_next(request)
            response.headers["X-Auth-Mode"] = "disabled"
            return response
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return JSONResponse(status_code=401, content={"error": "missing_bearer_token"})
        try:
            claims = verify_jwt(auth_header.split(" ", 1)[1].strip(), config=self.config)
        except AuthError as exc:
            return JSONResponse(status_code=exc.status_code, content={"error": exc.code, "detail": exc.detail})
        request.state.user_id = str(claims["sub"])
        request.state.jwt_claims = claims
        response = await call_next(request)
        response.headers["X-Auth-Mode"] = "jwt"
        response.headers["X-Auth-User"] = str(claims["sub"])
        return response


def install_auth(app: FastAPI, config: AuthConfig | None = None) -> AuthConfig:
    active = config or AuthConfig.from_env()
    app.add_middleware(AuthMiddleware, config=active)
    return active


def auth_status(config: AuthConfig | None = None) -> dict[str, Any]:
    active = config or AuthConfig.from_env()
    payload = asdict(active)
    payload["jwt_secret"] = bool(active.jwt_secret)
    payload["token_issue_key"] = bool(active.token_issue_key)
    return payload


def validate_token_issue_request(config: AuthConfig, supplied_key: str | None) -> None:
    if config.enabled and not config.allow_token_issue:
        raise AuthError("token_issue_disabled", "Token issuance is disabled in production mode.", 403)
    if config.token_issue_key and not hmac.compare_digest(supplied_key or "", config.token_issue_key):
        raise AuthError("invalid_token_issue_key", "Token issuance key is invalid.", 403)

