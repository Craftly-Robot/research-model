"""Runtime configuration helpers for consolidated Craftly modules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.craftly.shared.config_contracts import load_active_model_contract

ROOT = Path(__file__).resolve().parents[3]


def active_profile_path() -> Path:
    override = os.environ.get("CRAFTLY_ACTIVE_MODEL_PROFILE_PATH") or os.environ.get("CRAFTLY_ACTIVE_MODEL_PROFILE_PATH")
    return Path(override).expanduser().resolve() if override else ROOT / "config" / "active_model_profile.json"


def load_active_profile() -> dict[str, Any]:
    path = active_profile_path()
    if not path.exists():
        return {}
    try:
        return load_active_model_contract(path).model_dump()
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


