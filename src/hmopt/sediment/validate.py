"""Validate sediment candidates against the hub JSON-Schemas.

A candidate is `{"schema": <name>, "record": {...}}`. Validating here means the
write path can only emit records the hub `lint.py` will also accept — the
sediment output is lint-clean by construction.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None  # type: ignore[assignment]

from hmopt.skillhub.resolver import find_hub_root


@functools.lru_cache(maxsize=None)
def _load_schema(hub_root: str, name: str) -> dict[str, Any]:
    with open(Path(hub_root) / "schemas" / f"{name}.schema.json", encoding="utf-8") as f:
        return json.load(f)


def validate_candidate(candidate: dict[str, Any], hub_root: str | Path) -> list[str]:
    """Return a list of error strings ([] == valid)."""
    if jsonschema is None:  # pragma: no cover
        return ["jsonschema not installed"]
    schema_name = candidate.get("schema")
    record = candidate.get("record")
    if not schema_name or not isinstance(record, dict):
        return ["candidate missing 'schema' or 'record'"]
    hub = find_hub_root(hub_root) or Path(hub_root)
    try:
        schema = _load_schema(str(hub), str(schema_name))
    except OSError:
        return [f"unknown schema {schema_name!r}"]
    errs: list[str] = []
    validator = jsonschema.Draft7Validator(schema)
    for e in sorted(validator.iter_errors(record), key=lambda x: list(x.absolute_path)):
        loc = ".".join(str(p) for p in e.absolute_path) or "(root)"
        errs.append(f"{schema_name}@{loc}: {e.message}")
    return errs
