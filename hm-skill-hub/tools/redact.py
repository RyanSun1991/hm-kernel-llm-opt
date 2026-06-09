"""Secret-pattern scanner for hub Markdown content.

Conservative regex sweep for the secrets most likely to leak from kernel /
flash / test workflows: device serials, /dev/serial paths, ssh-style keys,
generic hex keys >= 40 chars, AKIA-style tokens.

Exit codes:
    0 — no hits
    1 — hits found (prints file:line:pattern)
    2 — usage error

CLI:
    python tools/redact.py [--check] [path ...]
    --check : exit non-zero on any hit (CI mode)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HUB = Path(__file__).resolve().parent.parent

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws-akid",         re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("ssh-priv",         re.compile(r"-----BEGIN (?:RSA|OPENSSH|DSA|EC|PGP) PRIVATE KEY-----")),
    ("generic-hex-key",  re.compile(r"\b[0-9a-fA-F]{40,}\b")),
    ("device-serial",    re.compile(r"\b(?:serial|SN|imei|IMEI)\s*[:=]\s*[A-Za-z0-9]{8,}\b")),
    ("dev-serial-path",  re.compile(r"/dev/(?:ttyUSB|ttyACM|serial/by-id/)[A-Za-z0-9_./-]+")),
    ("github-pat",       re.compile(r"\bghp_[A-Za-z0-9]{30,}\b")),
    ("slack-token",      re.compile(r"\bxox[abps]-[A-Za-z0-9-]{10,}\b")),
]

ALLOW_TAG = re.compile(r"\[REDACTED\]|\[FAKE\]|<!--\s*allow-secret\s*-->")


def scan_file(p: Path) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return hits
    for lineno, line in enumerate(text.splitlines(), 1):
        if ALLOW_TAG.search(line):
            continue
        for name, pat in PATTERNS:
            if pat.search(line):
                hits.append((lineno, name, line.strip()[:160]))
    return hits


def discover(root: Path) -> list[Path]:
    skip_dirs = {".git", "node_modules", "__pycache__"}
    skip_suffixes = {".schema.json"}
    out: list[Path] = []
    for p in root.rglob("*"):
        if any(part in skip_dirs for part in p.parts):
            continue
        if not p.is_file():
            continue
        if p.suffix.lower() in skip_suffixes:
            continue
        if p.suffix.lower() in {".md", ".yaml", ".yml", ".json", ".jsonl", ".txt"}:
            out.append(p)
    return out


def main(argv: list[str]) -> int:
    check = "--check" in argv
    paths = [Path(a) for a in argv if a != "--check"]
    roots = [p.resolve() for p in paths] or [HUB]
    total = 0
    for r in roots:
        targets = discover(r) if r.is_dir() else [r]
        for t in targets:
            for lineno, name, snippet in scan_file(t):
                print(f"{t}:{lineno}: secret-pattern={name}  | {snippet}")
                total += 1
    if total == 0:
        print(f"OK — no secret patterns matched in {len(roots)} root(s).")
        return 0
    sys.stderr.write(f"\n{total} potential secret(s) found.\n")
    return 1 if check else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
