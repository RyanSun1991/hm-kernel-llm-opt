"""Flamegraph / HiPerf report pre-processing.

`flamegraph_parser` consumes a normalized JSON payload (process/thread/symbol
maps + samples). Raw HiPerf artifacts come either already as that JSON, or as a
self-contained ``*_hiperfReport.html`` page that embeds the same payload as an
inline ``<script>`` blob. This module normalizes both into a JSON file on disk
and returns ``(source_path, json_output_path)`` pairs.

Design note: this used to be an un-committed local module, which made the whole
``hmopt`` package fail to import (the parser imports it at module load). It is
now a real, committed module so the package is importable from a clean checkout.
When an HTML report cannot be normalized it raises a clear error rather than
returning an empty result, so a missing/garbled report is visible instead of
silently yielding "no hotspots".
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# HiPerf report pages embed the sample payload as a JS assignment, e.g.
#   var sampleInfo = {...};   /   window.flamegraph = {...};
# We pull the first balanced JSON object following such an assignment.
_HTML_PAYLOAD_RE = re.compile(
    r"(?:sampleInfo|flameGraph|flamegraph|reportData|__DATA__)\s*=\s*",
    re.IGNORECASE,
)


def _extract_balanced_json(text: str, start: int) -> str | None:
    """Return the balanced ``{...}`` object that begins at/after ``start``."""
    open_idx = text.find("{", start)
    if open_idx < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(open_idx, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx : i + 1]
    return None


def _html_to_payload(html: str) -> dict | None:
    for m in _HTML_PAYLOAD_RE.finditer(html):
        blob = _extract_balanced_json(html, m.end())
        if not blob:
            continue
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def preprocess_flamegraph_file(
    path: str | Path,
    out_dir: str | Path,
    tags: Iterable[str] | None = None,
) -> tuple[Path, Path]:
    """Normalize a single flamegraph artifact into a JSON file on disk.

    Returns ``(source_path, json_output_path)``. JSON inputs are passed through
    unchanged (the parser reads the source directly); HTML inputs are converted
    to ``<out_dir>/<stem>.json``. Raises ``ValueError`` if an HTML report carries
    no recognizable embedded payload.
    """
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        return path, path

    if path.suffix.lower() in (".html", ".htm"):
        payload = _html_to_payload(path.read_text(encoding="utf-8", errors="replace"))
        if payload is None:
            raise ValueError(
                f"no embedded flamegraph payload found in HTML report: {path}"
            )
        json_path = out_dir / f"{path.stem}.json"
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info("Normalized HTML flamegraph report -> %s", json_path)
        return path, json_path

    # Unknown extension: treat as already-JSON content (best effort, loud on fail).
    json.loads(path.read_text(encoding="utf-8"))  # raises on non-JSON, by design
    return path, path


def preprocess_flamegraph_dir(
    input_dir: str | Path,
    out_dir: str | Path,
    tags: Iterable[str] | None = None,
) -> list[tuple[Path, Path]]:
    """Normalize every HiPerf report HTML under ``input_dir`` into JSON.

    Returns a list of ``(html_path, json_output_path)`` pairs. Reports that
    cannot be normalized are logged and skipped (the directory scan stays
    resilient), but an individual file converted via
    :func:`preprocess_flamegraph_file` still raises on malformed input.
    """
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    outputs: list[tuple[Path, Path]] = []
    for html in sorted(input_dir.rglob("*__sysmgr_hiperfReport.html")):
        try:
            outputs.append(preprocess_flamegraph_file(html, out_dir, tags))
        except ValueError as exc:
            logger.warning("Skipping unparseable HiPerf report %s: %s", html, exc)
    return outputs
