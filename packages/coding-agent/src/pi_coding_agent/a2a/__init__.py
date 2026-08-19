"""Bundled Tau A2A extension package."""
from __future__ import annotations

import os
from pathlib import Path


def is_enabled() -> bool:
    return os.environ.get("TAU_A2A_DISABLED", "").strip().lower() not in {"1", "true", "yes", "on"}


def builtin_extension_path() -> str:
    return str(Path(__file__).with_name("extension.py"))


__all__ = ["builtin_extension_path", "is_enabled"]
