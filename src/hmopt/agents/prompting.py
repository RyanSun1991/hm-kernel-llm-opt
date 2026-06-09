"""Helpers for loading and rendering agent prompt templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _normalize_prompt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "\n".join(str(item) for item in value)
    return str(value)


def load_prompt_template(template_name: str, fallback: str) -> str:
    path = _PROMPTS_DIR / template_name
    if not path.exists():
        return fallback.strip()
    return path.read_text(encoding="utf-8").strip()


def render_prompt_template(template_name: str, fallback: str, **kwargs: Any) -> str:
    template = load_prompt_template(template_name, fallback)
    values = {key: _normalize_prompt_value(value) for key, value in kwargs.items()}
    return template.format_map(_SafeFormatDict(values)).strip()
