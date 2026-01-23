"""Kernel code ingestion using clangd or regex fallback."""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from hmopt.analysis.static.indexer import index_repo

from .clangd_client import ClangdClient, ClangdConfig

logger = logging.getLogger(__name__)


SYMBOL_KIND_NAMES = {
    1: "file",
    2: "module",
    3: "namespace",
    4: "package",
    5: "class",
    6: "method",
    7: "property",
    8: "field",
    9: "constructor",
    10: "enum",
    11: "interface",
    12: "function",
    13: "variable",
    14: "constant",
    15: "string",
    16: "number",
    17: "boolean",
    18: "array",
    19: "object",
    20: "key",
    21: "null",
    22: "enum_member",
    23: "struct",
    24: "event",
    25: "operator",
    26: "type_parameter",
}
SYMBOL_KIND_BY_NAME = {name: kind for kind, name in SYMBOL_KIND_NAMES.items()}

DEFAULT_SYMBOL_KINDS = [
    "function",
    "method",
    "constructor",
    "class",
    "struct",
    "enum",
    "enum_member",
    "interface",
    "type_parameter",
    "variable",
    "constant",
    "field",
    "property",
]

KIND_FUNCTIONS = {"function", "method", "constructor"}
KIND_TYPES = {"class", "struct", "enum", "enum_member", "interface", "type_parameter"}
KIND_FIELDS = {"field", "property"}
KIND_VARIABLES = {"variable"}
KIND_CONSTANTS = {"constant"}

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class SymbolRecord:
    symbol_id: str
    name: str
    qualname: str
    kind_id: int
    kind: str
    path: Path
    start_line: int
    end_line: int
    start_char: int
    selection_line: int
    selection_char: int
    detail: Optional[str]
    container: Optional[str]


@dataclass
class CodeChunk:
    path: Path
    symbol_id: str
    symbol_name: str
    symbol_qualname: str
    kind: str
    start_line: int
    end_line: int
    text: str
    parser: str
    detail: Optional[str] = None
    container: Optional[str] = None


@dataclass
class CodeRelation:
    src_id: str
    dst_id: str
    kind: str
    src_name: str
    dst_name: str
    src_kind: str
    dst_kind: str
    src_path: Optional[str] = None
    dst_path: Optional[str] = None


@dataclass
class FileSummary:
    path: Path
    text: str


@dataclass
class RelationSummary:
    symbol_id: str
    symbol_name: str
    symbol_kind: str
    path: str
    text: str


@dataclass
class CodeIndex:
    chunks: list[CodeChunk]
    relations: list[CodeRelation]
    file_summaries: list[FileSummary]
    relation_summaries: list[RelationSummary]


def _symbol_kind_name(kind_id: int) -> str:
    return SYMBOL_KIND_NAMES.get(kind_id, f"kind_{kind_id}")


def _resolve_symbol_kinds(config: Optional[ClangdConfig]) -> set[int]:
    if not config or not config.symbol_kinds:
        return {SYMBOL_KIND_BY_NAME[name] for name in DEFAULT_SYMBOL_KINDS}
    resolved: set[int] = set()
    for name in config.symbol_kinds:
        if isinstance(name, int):
            resolved.add(name)
            continue
        key = str(name).strip().lower()
        kind_id = SYMBOL_KIND_BY_NAME.get(key)
        if kind_id:
            resolved.add(kind_id)
    if not resolved:
        resolved = {SYMBOL_KIND_BY_NAME[name] for name in DEFAULT_SYMBOL_KINDS}
    return resolved


def _symbol_id(path: Path, qualname: str, start_line: int, kind: str) -> str:
    return f"{path}:{qualname}:{start_line + 1}:{kind}"


def _qualname(name: str, parent: Optional[SymbolRecord]) -> str:
    if parent and parent.qualname:
        return f"{parent.qualname}::{name}"
    return name


def _load_compile_commands(path: Path) -> list[Path]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    files: list[Path] = []
    for entry in data:
        file_path = entry.get("file")
        if file_path:
            files.append(Path(file_path))
    return files


def _flatten_symbols(
    symbols: list[dict],
    path: Path,
    parent: Optional[SymbolRecord],
    records: list[SymbolRecord],
    relations: dict[tuple[str, str, str], CodeRelation],
) -> None:
    for sym in symbols:
        name = sym.get("name") or ""
        if not name:
            continue
        kind_id = int(sym.get("kind", 0) or 0)
        kind_name = _symbol_kind_name(kind_id)
        rng = sym.get("range", {}) or {}
        start = rng.get("start", {}) or {}
        end = rng.get("end", {}) or {}
        start_line = int(start.get("line", 0))
        end_line = int(end.get("line", start_line))
        start_char = int(start.get("character", 0))
        sel = sym.get("selectionRange", rng) or {}
        sel_start = sel.get("start", {}) or {}
        selection_line = int(sel_start.get("line", start_line))
        selection_char = int(sel_start.get("character", start_char))
        detail = sym.get("detail")
        qualname = _qualname(name, parent)
        sym_id = _symbol_id(path, qualname, start_line, kind_name)
        record = SymbolRecord(
            symbol_id=sym_id,
            name=name,
            qualname=qualname,
            kind_id=kind_id,
            kind=kind_name,
            path=path,
            start_line=start_line,
            end_line=end_line,
            start_char=start_char,
            selection_line=selection_line,
            selection_char=selection_char,
            detail=detail,
            container=parent.qualname if parent else None,
        )
        records.append(record)
        if parent:
            _add_relation(
                relations,
                src=parent,
                dst=record,
                kind="contains",
            )
        children = sym.get("children", []) or []
        _flatten_symbols(children, path, record, records, relations)


def _slice_lines(text: str, start: int, end: int) -> str:
    lines = text.splitlines()
    start = max(start, 0)
    end = min(end, len(lines) - 1)
    return "\n".join(lines[start : end + 1])


def _clangd_available(binary: str) -> bool:
    return shutil.which(binary) is not None


def _scan_tokens(text: str, names: set[str], max_hits: int) -> list[str]:
    hits: list[str] = []
    seen: set[str] = set()
    for match in TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        if token in names and token not in seen:
            hits.append(token)
            seen.add(token)
            if max_hits and len(hits) >= max_hits:
                break
    return hits


def _limit_names(names: list[str], limit: int) -> set[str]:
    if limit and len(names) > limit:
        logger.warning("Usage scan names truncated: %d -> %d", len(names), limit)
        names = sorted(names, key=len, reverse=True)[:limit]
    return set(names)


def _add_relation(
    relations: dict[tuple[str, str, str], CodeRelation],
    *,
    src: SymbolRecord,
    dst: Optional[SymbolRecord],
    kind: str,
    dst_name: Optional[str] = None,
    dst_path: Optional[Path] = None,
    dst_kind: Optional[str] = None,
) -> None:
    if dst:
        dst_id = dst.symbol_id
        dst_name = dst.name
        dst_kind = dst.kind
        dst_path_str = str(dst.path)
    else:
        dst_name = dst_name or "external"
        dst_id = f"external::{dst_name}"
        dst_kind = dst_kind or "external"
        dst_path_str = str(dst_path) if dst_path else None
    key = (src.symbol_id, dst_id, kind)
    if key in relations:
        return
    relations[key] = CodeRelation(
        src_id=src.symbol_id,
        dst_id=dst_id,
        kind=kind,
        src_name=src.name,
        dst_name=dst_name,
        src_kind=src.kind,
        dst_kind=dst_kind,
        src_path=str(src.path),
        dst_path=dst_path_str,
    )


def _build_file_summaries(
    records: list[SymbolRecord],
    *,
    max_items: int,
) -> list[FileSummary]:
    summaries: list[FileSummary] = []
    by_file: dict[Path, list[SymbolRecord]] = defaultdict(list)
    for rec in records:
        by_file[rec.path].append(rec)
    for path, symbols in by_file.items():
        kinds = Counter(sym.kind for sym in symbols)
        functions = sorted({sym.name for sym in symbols if sym.kind in KIND_FUNCTIONS})
        types = sorted({sym.name for sym in symbols if sym.kind in KIND_TYPES})
        globals_ = sorted({sym.name for sym in symbols if sym.kind in KIND_VARIABLES})
        constants = sorted({sym.name for sym in symbols if sym.kind in KIND_CONSTANTS})
        fields = sorted({sym.name for sym in symbols if sym.kind in KIND_FIELDS})
        text = (
            f"file summary: {path}\n"
            f"counts: {dict(kinds)}\n"
            f"functions: {functions[:max_items]}\n"
            f"types: {types[:max_items]}\n"
            f"globals: {globals_[:max_items]}\n"
            f"constants: {constants[:max_items]}\n"
            f"fields: {fields[:max_items]}\n"
        )
        summaries.append(FileSummary(path=path, text=text))
    return summaries


def _build_relation_summaries(
    records: list[SymbolRecord],
    relations: list[CodeRelation],
    *,
    max_items: int,
) -> list[RelationSummary]:
    summaries: list[RelationSummary] = []
    rel_map: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for rel in relations:
        rel_map[rel.src_id][rel.kind].append(rel.dst_name)

    record_map = {rec.symbol_id: rec for rec in records}
    for sym_id, kinds in rel_map.items():
        rec = record_map.get(sym_id)
        if not rec:
            continue
        if rec.kind not in KIND_FUNCTIONS and rec.kind not in KIND_TYPES:
            continue
        lines = [
            f"symbol relations: {rec.qualname} ({rec.kind})",
            f"path: {rec.path}",
        ]
        for kind, targets in sorted(kinds.items()):
            uniq = sorted(set(targets))
            lines.append(f"{kind}: {uniq[:max_items]}")
        text = "\n".join(lines)
        summaries.append(
            RelationSummary(
                symbol_id=rec.symbol_id,
                symbol_name=rec.qualname,
                symbol_kind=rec.kind,
                path=str(rec.path),
                text=text,
            )
        )
    return summaries


def _fallback_index(repo_path: Path) -> CodeIndex:
    chunks: list[CodeChunk] = []
    symbols = index_repo(repo_path)
    for sym in symbols:
        try:
            text = sym.file_path.read_text(encoding="utf-8", errors="ignore")
        except FileNotFoundError:
            continue
        snippet = _slice_lines(text, sym.line - 1, sym.line + 25)
        sym_id = _symbol_id(sym.file_path, sym.name, sym.line - 1, sym.kind)
        chunks.append(
            CodeChunk(
                path=sym.file_path,
                symbol_id=sym_id,
                symbol_name=sym.name,
                symbol_qualname=sym.name,
                kind=sym.kind,
                start_line=sym.line,
                end_line=min(sym.line + 25, len(text.splitlines())),
                text=snippet,
                parser="regex",
            )
        )
    logger.info("Fallback indexer produced chunks=%d", len(chunks))
    return CodeIndex(chunks=chunks, relations=[], file_summaries=[], relation_summaries=[])


def index_kernel_code(
    repo_path: Path,
    *,
    clangd_config: Optional[ClangdConfig],
    max_files: int = 5000,
    files: Optional[Iterable[Path]] = None,
) -> CodeIndex:
    if not clangd_config or not clangd_config.compile_commands_dir:
        logger.warning("clangd config missing, fallback to regex indexer")
        return _fallback_index(repo_path)
    if not _clangd_available(clangd_config.binary):
        logger.warning("clangd binary not available: %s", clangd_config.binary)
        return _fallback_index(repo_path)

    compile_commands = Path(clangd_config.compile_commands_dir) / "compile_commands.json"
    if files is None:
        files = _load_compile_commands(compile_commands)
        if not files:
            logger.warning("compile_commands.json empty or missing, fallback to regex")
            return _fallback_index(repo_path)
    else:
        resolved: list[Path] = []
        for path in files:
            p = Path(path)
            if not p.is_absolute():
                p = (repo_path / p).resolve()
            resolved.append(p)
        files = resolved
    files = list(files)[:max_files]
    client = ClangdClient(clangd_config, repo_path)
    records: list[SymbolRecord] = []
    relations: dict[tuple[str, str, str], CodeRelation] = {}
    file_texts: dict[Path, str] = {}
    try:
        for path in files:
            if not path.exists():
                continue
            try:
                symbols = client.document_symbols(path)
            except Exception as exc:
                logger.warning("clangd documentSymbol failed: path=%s err=%s", path, exc)
                continue
            _flatten_symbols(symbols, path, None, records, relations)
            if path not in file_texts:
                file_texts[path] = path.read_text(encoding="utf-8", errors="ignore")

        if not records:
            logger.warning("clangd returned no symbols, fallback to regex")
            return _fallback_index(repo_path)

        allowed_kinds = _resolve_symbol_kinds(clangd_config)
        symbol_by_id = {rec.symbol_id: rec for rec in records}
        symbol_by_name: dict[str, list[SymbolRecord]] = defaultdict(list)
        symbol_by_name_path: dict[tuple[str, Path], list[SymbolRecord]] = defaultdict(list)
        for rec in records:
            symbol_by_name[rec.name].append(rec)
            symbol_by_name_path[(rec.name, rec.path)].append(rec)

        chunks: list[CodeChunk] = []
        for rec in records:
            if rec.kind_id not in allowed_kinds:
                continue
            text = file_texts.get(rec.path)
            if text is None:
                try:
                    text = rec.path.read_text(encoding="utf-8", errors="ignore")
                except FileNotFoundError:
                    continue
                file_texts[rec.path] = text
            snippet = _slice_lines(text, rec.start_line, rec.end_line)
            if not snippet.strip():
                continue
            chunks.append(
                CodeChunk(
                    path=rec.path,
                    symbol_id=rec.symbol_id,
                    symbol_name=rec.name,
                    symbol_qualname=rec.qualname,
                    kind=rec.kind,
                    start_line=rec.start_line + 1,
                    end_line=rec.end_line + 1,
                    text=snippet,
                    parser="clangd",
                    detail=rec.detail,
                    container=rec.container,
                )
            )

        relation_limit = clangd_config.relation_max_per_symbol
        relation_counts: dict[str, int] = defaultdict(int)

        def _add_limited_relation(
            src: SymbolRecord,
            dst: Optional[SymbolRecord],
            kind: str,
            dst_name: Optional[str] = None,
            dst_path: Optional[Path] = None,
            dst_kind: Optional[str] = None,
        ) -> None:
            if relation_limit and relation_counts[src.symbol_id] >= relation_limit:
                return
            before = len(relations)
            _add_relation(
                relations,
                src=src,
                dst=dst,
                kind=kind,
                dst_name=dst_name,
                dst_path=dst_path,
                dst_kind=dst_kind,
            )
            if len(relations) > before:
                relation_counts[src.symbol_id] += 1

        type_names = sorted({rec.name for rec in records if rec.kind in KIND_TYPES})
        macro_names = sorted(
            {rec.name for rec in records if rec.kind in KIND_CONSTANTS or rec.kind in KIND_VARIABLES}
        )
        function_names = sorted({rec.name for rec in records if rec.kind in KIND_FUNCTIONS})
        if clangd_config.usage_scan_enabled:
            type_name_set = _limit_names(type_names, clangd_config.usage_scan_max_names)
            macro_name_set = _limit_names(macro_names, clangd_config.usage_scan_max_names)
            function_name_set = _limit_names(function_names, clangd_config.usage_scan_max_names)
        else:
            type_name_set = set()
            macro_name_set = set()
            function_name_set = set()

        def _resolve_by_name(name: str, path: Path) -> Optional[SymbolRecord]:
            candidates = symbol_by_name_path.get((name, path))
            if candidates:
                return candidates[0]
            candidates = symbol_by_name.get(name)
            if candidates:
                return candidates[0]
            return None

        call_edge_sources: set[str] = set()
        # Call hierarchy (clangd), optional.
        if clangd_config.call_hierarchy_enabled:
            max_functions = clangd_config.call_hierarchy_max_functions
            max_calls = clangd_config.call_hierarchy_max_calls
            max_depth = clangd_config.call_hierarchy_max_depth
            function_records = [rec for rec in records if rec.kind in KIND_FUNCTIONS]
            for rec in function_records[:max_functions]:
                if relation_limit and relation_counts[rec.symbol_id] >= relation_limit:
                    continue
                try:
                    items = client.prepare_call_hierarchy(rec.path, rec.selection_line, rec.selection_char)
                except Exception as exc:
                    logger.debug("clangd callHierarchy prepare failed: %s", exc)
                    continue
                seen: set[tuple[str, str]] = set()

                def _collect_calls(item: dict, depth: int) -> list[tuple[str, Optional[Path]]]:
                    if depth > max_depth:
                        return []
                    results: list[tuple[str, Optional[Path]]] = []
                    try:
                        outgoing = client.outgoing_calls(item)
                    except Exception:
                        return results
                    for call in outgoing:
                        target = call.get("to") or {}
                        name = target.get("name")
                        uri = target.get("uri")
                        path = Path(uri.replace("file://", "")) if uri and uri.startswith("file://") else None
                        if not name:
                            continue
                        key = (name, str(path) if path else "")
                        if key in seen:
                            continue
                        seen.add(key)
                        results.append((name, path))
                        if depth < max_depth:
                            results.extend(_collect_calls(target, depth + 1))
                        if max_calls and len(results) >= max_calls:
                            break
                    return results

                for item in items:
                    targets = _collect_calls(item, 1)
                    for name, path in targets[:max_calls] if max_calls else targets:
                        dst = _resolve_by_name(name, path or rec.path)
                        _add_limited_relation(rec, dst, "calls", dst_name=name, dst_path=path)
                        call_edge_sources.add(rec.symbol_id)

        # Token-based usage scan for types/macros/functions.
        for chunk in chunks:
            rec = symbol_by_id.get(chunk.symbol_id)
            if not rec:
                continue
            if rec.kind not in KIND_FUNCTIONS:
                continue
            if clangd_config.usage_scan_enabled:
                used_types = _scan_tokens(chunk.text, type_name_set, clangd_config.relation_max_per_symbol)
                for name in used_types:
                    dst = _resolve_by_name(name, rec.path)
                    _add_limited_relation(rec, dst, "uses_type", dst_name=name, dst_path=dst.path if dst else None)
                used_macros = _scan_tokens(chunk.text, macro_name_set, clangd_config.relation_max_per_symbol)
                for name in used_macros:
                    dst = _resolve_by_name(name, rec.path)
                    _add_limited_relation(rec, dst, "uses_macro", dst_name=name, dst_path=dst.path if dst else None)
            # Regex fallback for call edges when call hierarchy is disabled or empty.
            if not clangd_config.call_hierarchy_enabled or rec.symbol_id not in call_edge_sources:
                called = _scan_tokens(chunk.text, function_name_set, clangd_config.relation_max_per_symbol)
                for name in called:
                    if name == rec.name:
                        continue
                    dst = _resolve_by_name(name, rec.path)
                    _add_limited_relation(rec, dst, "calls", dst_name=name, dst_path=dst.path if dst else None)

        file_summaries: list[FileSummary] = []
        if clangd_config.file_summary_enabled:
            file_summaries = _build_file_summaries(
                records,
                max_items=clangd_config.relation_summary_max_items,
            )
        relation_summaries: list[RelationSummary] = []
        if clangd_config.relation_summary_enabled:
            relation_summaries = _build_relation_summaries(
                records,
                list(relations.values()),
                max_items=clangd_config.relation_summary_max_items,
            )

        logger.info(
            "clangd indexer produced chunks=%d symbols=%d relations=%d",
            len(chunks),
            len(records),
            len(relations),
        )
        if not chunks:
            return _fallback_index(repo_path)
        return CodeIndex(
            chunks=chunks,
            relations=list(relations.values()),
            file_summaries=file_summaries,
            relation_summaries=relation_summaries,
        )
    finally:
        client.close()
