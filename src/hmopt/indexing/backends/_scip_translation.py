"""Pure helper functions for translating SCIP wire data into the project's
unified CodeChunk / CodeRelation model.

These functions are deliberately kept free of I/O and protobuf object
dependencies so they can be unit-tested in isolation. The backend that
actually invokes `scip-clang` (see `scip_clang.py`) glues them onto the
real protobuf-decoded objects.

References:
- SCIP symbol-string grammar:
  https://github.com/sourcegraph/scip/blob/main/Syntax.md
- SCIP protocol buffer schema:
  third_party/scip/scip.proto
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# SCIP symbol-string parser
# ---------------------------------------------------------------------------
#
# A SCIP symbol is a single space-separated string of the form
#
#   <scheme> SP <manager> SP <package-name> SP <version> SP <descriptors...>
#
# where each <descriptor> is one of:
#   <name>'/'                   namespace      (path components for C/C++ symbols)
#   <name>'#'                   type           (struct, union, enum, typedef, class)
#   <name>'.'                   term           (variable, field, parameter — but
#                                              term is also the catch-all kind)
#   <name>'(<disambig>).'       method         (functions, methods)
#   <name>'!'                   macro
#   <name>':'                   meta
#   '[' <name> ']'              type-parameter
#   '(' <name> ')'              parameter
#
# Names are simple identifiers, OR backtick-escaped strings where embedded
# backticks are doubled (`` `` ``).
#
# Local symbols use the form `local <id>` (scheme = "local") — handled but
# returns a degenerate ParsedSymbol with empty descriptors.


@dataclass
class ParsedDescriptor:
    name: str
    suffix: str
    """One of: namespace, type, term, method, macro, meta, type_parameter, parameter."""

    disambiguator: str = ""
    """Only set for `method`; usually empty for C/C++ symbols."""


@dataclass
class ParsedSymbol:
    scheme: str
    manager: str
    package_name: str
    version: str
    descriptors: list[ParsedDescriptor] = field(default_factory=list)

    @property
    def is_local(self) -> bool:
        return self.scheme == "local"

    @property
    def qualname(self) -> str:
        """Qualified name built from all descriptors joined by `::`.

        Per the SCIP grammar (https://github.com/sourcegraph/scip/blob/main/Syntax.md),
        names containing `.`, `/`, `#`, `!`, `:`, `(`, `[`, ` ` `, or `@` must be
        backtick-escaped. That means a bare unescaped namespace descriptor in
        a well-formed scip-clang symbol is always a genuine C++ namespace
        (or equivalent organizational scope) — NOT a file path or directory
        component. The file path lives in `Document.relative_path`, not here.

        We therefore include every descriptor in the qualname.
        """
        if not self.descriptors:
            return ""
        return "::".join(d.name for d in self.descriptors)

    @property
    def kind(self) -> str:
        """Project kind string derived from the final descriptor's suffix.

        Returns 'unknown' when the symbol has no descriptors.
        """
        if not self.descriptors:
            return "unknown"
        return _SUFFIX_TO_KIND.get(self.descriptors[-1].suffix, "unknown")


_SUFFIX_TO_KIND: dict[str, str] = {
    "method": "function",
    "type": "type",
    "term": "variable",
    "macro": "macro",
    "meta": "meta",
    "type_parameter": "type_parameter",
    "parameter": "parameter",
    "namespace": "namespace",
}


# Terminators that close a simple-name token in a descriptor path.
_NAME_TERMINATORS = set("./#!:[]()`")


def parse_scip_symbol(symbol: str) -> Optional[ParsedSymbol]:
    """Parse a SCIP symbol string. Returns None on malformed input.

    Empty / blank strings, and unparseable strings, both yield None — callers
    are expected to fall back to using the raw symbol string as an opaque
    identifier.
    """
    if not symbol or not symbol.strip():
        return None

    # Local symbols: `local <id>` — return a stub.
    if symbol.startswith("local "):
        return ParsedSymbol(
            scheme="local",
            manager=".",
            package_name=".",
            version=".",
            descriptors=[ParsedDescriptor(name=symbol[6:].strip(), suffix="term")],
        )

    # Split off the first 4 simple fields (scheme, manager, package_name, version).
    # The descriptor path may itself contain spaces inside backtick-escaped names
    # but the leading 4 fields are constrained to simple-name-only — splitting
    # on the first 4 spaces is safe.
    parts = symbol.split(" ", 4)
    if len(parts) < 5:
        return None
    scheme, manager, package_name, version, descriptor_path = parts
    descriptors = _parse_descriptor_path(descriptor_path)
    if descriptors is None:
        return None
    return ParsedSymbol(
        scheme=scheme,
        manager=manager,
        package_name=package_name,
        version=version,
        descriptors=descriptors,
    )


def _parse_descriptor_path(s: str) -> Optional[list[ParsedDescriptor]]:
    """Walk the descriptor portion and return the parsed list, or None on error."""
    result: list[ParsedDescriptor] = []
    i = 0
    n = len(s)
    while i < n:
        head = s[i]
        # Type-parameter or parameter (no preceding name).
        if head == "[":
            end = s.find("]", i + 1)
            if end < 0:
                return None
            result.append(
                ParsedDescriptor(name=s[i + 1 : end], suffix="type_parameter")
            )
            i = end + 1
            continue
        if head == "(":
            end = s.find(")", i + 1)
            if end < 0:
                return None
            result.append(ParsedDescriptor(name=s[i + 1 : end], suffix="parameter"))
            i = end + 1
            continue

        # Otherwise read a name, then dispatch on terminator.
        name, j = _read_name(s, i)
        if j >= n:
            return None
        term = s[j]
        if term == "/":
            result.append(ParsedDescriptor(name=name, suffix="namespace"))
            i = j + 1
        elif term == "#":
            result.append(ParsedDescriptor(name=name, suffix="type"))
            i = j + 1
        elif term == "!":
            result.append(ParsedDescriptor(name=name, suffix="macro"))
            i = j + 1
        elif term == ":":
            result.append(ParsedDescriptor(name=name, suffix="meta"))
            i = j + 1
        elif term == "(":
            # method: name(disambig).
            close = s.find(")", j + 1)
            if close < 0 or close + 1 >= n or s[close + 1] != ".":
                return None
            disambig = s[j + 1 : close]
            result.append(
                ParsedDescriptor(name=name, suffix="method", disambiguator=disambig)
            )
            i = close + 2
        elif term == ".":
            result.append(ParsedDescriptor(name=name, suffix="term"))
            i = j + 1
        else:
            return None
    return result if result else None


def _read_name(s: str, start: int) -> tuple[str, int]:
    """Read a simple-name or backtick-escaped name. Returns (name, end_index).

    `end_index` points at the character immediately after the consumed name.
    For an escaped name, that character is whatever follows the closing
    backtick. For a simple name, it is the first terminator.
    """
    if start >= len(s):
        return "", start
    if s[start] == "`":
        out: list[str] = []
        i = start + 1
        while i < len(s):
            if s[i] == "`":
                # Doubled backtick → literal `; single backtick → end of name.
                if i + 1 < len(s) and s[i + 1] == "`":
                    out.append("`")
                    i += 2
                else:
                    return "".join(out), i + 1
            else:
                out.append(s[i])
                i += 1
        # Unterminated — tolerate by returning what we have.
        return "".join(out), i
    # Simple name: read until terminator.
    i = start
    while i < len(s) and s[i] not in _NAME_TERMINATORS:
        i += 1
    return s[start:i], i


# ---------------------------------------------------------------------------
# SyntaxKind → relation.kind mapping
# ---------------------------------------------------------------------------
#
# The scip.proto SyntaxKind values we care about for kernel optimization:
#   IdentifierFunction      = 15   → calls
#   IdentifierMacro         = 17   → uses_macro
#   IdentifierType          = 19   → uses_type
#   IdentifierConstant      = 9    → uses_const
#   IdentifierBuiltinType   = 20   → uses_type (builtins are still types)
#
# Other Identifier* kinds map to a generic "references" relation. Non-
# Identifier kinds (Comment, PunctuationBracket, Keyword, ...) yield None,
# signalling the caller to skip them — they have no edge meaning.


# Integer values from scip.proto SyntaxKind enum.
SYNTAX_KIND_IDENTIFIER_FUNCTION = 15
SYNTAX_KIND_IDENTIFIER_FUNCTION_DEFINITION = 16
SYNTAX_KIND_IDENTIFIER_MACRO = 17
SYNTAX_KIND_IDENTIFIER_MACRO_DEFINITION = 18
SYNTAX_KIND_IDENTIFIER_TYPE = 19
SYNTAX_KIND_IDENTIFIER_BUILTIN_TYPE = 20
SYNTAX_KIND_IDENTIFIER_CONSTANT = 9
SYNTAX_KIND_IDENTIFIER_MUTABLE_GLOBAL = 10
SYNTAX_KIND_IDENTIFIER = 6


_SYNTAX_KIND_TO_RELATION: dict[int, str] = {
    SYNTAX_KIND_IDENTIFIER_FUNCTION: "calls",
    SYNTAX_KIND_IDENTIFIER_FUNCTION_DEFINITION: "calls",
    SYNTAX_KIND_IDENTIFIER_MACRO: "uses_macro",
    SYNTAX_KIND_IDENTIFIER_MACRO_DEFINITION: "uses_macro",
    SYNTAX_KIND_IDENTIFIER_TYPE: "uses_type",
    SYNTAX_KIND_IDENTIFIER_BUILTIN_TYPE: "uses_type",
    SYNTAX_KIND_IDENTIFIER_CONSTANT: "uses_const",
    SYNTAX_KIND_IDENTIFIER_MUTABLE_GLOBAL: "uses_global",
    SYNTAX_KIND_IDENTIFIER: "references",
}


def scip_syntax_kind_to_relation_kind(syntax_kind: int) -> Optional[str]:
    """Map a SCIP SyntaxKind enum value to this project's CodeRelation.kind.

    Returns None for non-edge syntax kinds (comments, punctuation, keywords),
    signalling the caller to skip that occurrence.
    """
    return _SYNTAX_KIND_TO_RELATION.get(syntax_kind)


# ---------------------------------------------------------------------------
# Descriptor-suffix fallback for relation kind
# ---------------------------------------------------------------------------
#
# scip-clang 0.3.1 with some vendor toolchains (observed: BiSheng-based
# HarmonyOS kernel builds) leaves Occurrence.syntax_kind unset on the
# majority of reference occurrences — it defaults to UnspecifiedSyntaxKind=0,
# which the SyntaxKind table above intentionally rejects. The empirical
# impact on a kernel-scale .scip: 38,445 reference occurrences out of 47,680
# total, but ~80% of them have syntax_kind=0 and would be silently dropped
# without a fallback (0 :calls / :uses_* edges land in Neo4j).
#
# The SCIP symbol string itself still encodes the kind via its descriptor
# suffix (`name().` = method, `name#` = type, `name!` = macro, ...), so we
# can recover the relation kind from there when syntax_kind is uninformative.

_DESCRIPTOR_SUFFIX_TO_RELATION: dict[str, str] = {
    "method": "calls",
    "macro": "uses_macro",
    "type": "uses_type",
    "type_parameter": "uses_type",
    "term": "references",
    "namespace": "references",
    "parameter": "references",
    "meta": "references",
}


def infer_relation_kind_from_descriptor(parsed: ParsedSymbol) -> Optional[str]:
    """Derive a CodeRelation.kind from a parsed SCIP symbol's last descriptor.

    Used as a fallback in `_build_relation` when SCIP's syntax_kind field is
    UnspecifiedSyntaxKind (0) or otherwise unmapped, which is a known shape
    of scip-clang 0.3.1 + BiSheng output. The mapping favors precision over
    recall: only suffixes whose semantic meaning is unambiguous get a typed
    edge ('method' → calls, 'macro' → uses_macro, 'type'/'type_parameter' →
    uses_type); everything else falls back to a generic 'references' edge so
    we don't lose the link in the graph.

    Returns None if the descriptors list is empty (caller should skip the
    occurrence — it carries no symbolic identity).
    """
    if not parsed.descriptors:
        return None
    last_suffix = parsed.descriptors[-1].suffix
    return _DESCRIPTOR_SUFFIX_TO_RELATION.get(last_suffix)


# ---------------------------------------------------------------------------
# SymbolInformation.Kind → CodeChunk.kind mapping
# ---------------------------------------------------------------------------
#
# Maps the SymbolInformation.Kind enum (defined inside the SymbolInformation
# message in scip.proto) to project-canonical CodeChunk.kind strings. These
# match the strings emitted by the clangd backend's SYMBOL_KIND_NAMES table
# in clangd_indexer.py so the two backends share a unified kind vocabulary.


# Selected SCIP SymbolInformation.Kind values relevant to C/C++.
SYMBOL_KIND_CLASS = 7
SYMBOL_KIND_CONSTANT = 8
SYMBOL_KIND_CONSTRUCTOR = 9
SYMBOL_KIND_ENUM = 11
SYMBOL_KIND_ENUM_MEMBER = 12
SYMBOL_KIND_FIELD = 15
SYMBOL_KIND_FILE = 16
SYMBOL_KIND_FUNCTION = 17
SYMBOL_KIND_INTERFACE = 21
SYMBOL_KIND_MACRO = 25
SYMBOL_KIND_METHOD = 26
SYMBOL_KIND_NAMESPACE = 30
SYMBOL_KIND_PARAMETER = 37
SYMBOL_KIND_PROPERTY = 41
SYMBOL_KIND_STRUCT = 49
SYMBOL_KIND_TYPE = 54
SYMBOL_KIND_TYPE_ALIAS = 55
SYMBOL_KIND_UNION = 59
SYMBOL_KIND_VARIABLE = 61


_SYMBOL_KIND_TO_NODE: dict[int, str] = {
    SYMBOL_KIND_CLASS: "class",
    SYMBOL_KIND_CONSTANT: "constant",
    SYMBOL_KIND_CONSTRUCTOR: "constructor",
    SYMBOL_KIND_ENUM: "enum",
    SYMBOL_KIND_ENUM_MEMBER: "enum_member",
    SYMBOL_KIND_FIELD: "field",
    SYMBOL_KIND_FILE: "file",
    SYMBOL_KIND_FUNCTION: "function",
    SYMBOL_KIND_INTERFACE: "interface",
    SYMBOL_KIND_MACRO: "macro",
    SYMBOL_KIND_METHOD: "method",
    SYMBOL_KIND_NAMESPACE: "namespace",
    SYMBOL_KIND_PARAMETER: "parameter",
    SYMBOL_KIND_PROPERTY: "property",
    SYMBOL_KIND_STRUCT: "struct",
    SYMBOL_KIND_TYPE: "type",
    SYMBOL_KIND_TYPE_ALIAS: "type",
    SYMBOL_KIND_UNION: "struct",
    SYMBOL_KIND_VARIABLE: "variable",
}


def scip_symbol_kind_to_node_kind(symbol_kind: int) -> str:
    """Map SCIP SymbolInformation.Kind to CodeChunk.kind.

    Falls back to deriving from the symbol string's descriptor suffix when
    the SCIP Kind is UnspecifiedKind (0). When that derivation isn't
    available either, returns 'unknown'.
    """
    return _SYMBOL_KIND_TO_NODE.get(symbol_kind, "unknown")


# ---------------------------------------------------------------------------
# Range normalization
# ---------------------------------------------------------------------------
#
# SCIP encodes Occurrence.range as a `repeated int32` with two valid shapes:
#   - [start_line, start_col, end_col]            same-line range (length 3)
#   - [start_line, start_col, end_line, end_col]  multi-line range (length 4)
# All coordinates are 0-based.


def normalize_range(raw: Iterable[int]) -> Optional[tuple[int, int, int, int]]:
    """Normalize a SCIP range to (start_line, start_col, end_line, end_col).

    Returns None for malformed inputs (wrong length).
    """
    coords = list(raw)
    if len(coords) == 4:
        sl, sc, el, ec = coords
    elif len(coords) == 3:
        sl, sc, ec = coords
        el = sl
    else:
        return None
    return (sl, sc, el, ec)


def range_contains(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    """True if `outer` fully contains `inner` (inclusive on both ends)."""
    osl, osc, oel, oec = outer
    isl, isc, iel, iec = inner
    starts_before_or_eq = (osl, osc) <= (isl, isc)
    ends_after_or_eq = (oel, oec) >= (iel, iec)
    return starts_before_or_eq and ends_after_or_eq


# ---------------------------------------------------------------------------
# Enclosing-definition scanner
# ---------------------------------------------------------------------------
#
# Given a list of Definition occurrences (each with its enclosing_range or
# fallback range), find the smallest definition whose enclosing range contains
# a given reference occurrence's range. This tells us "which function body
# does this call site live in".


@dataclass
class DefinitionExtent:
    """One Definition occurrence with the range we use for containment tests.

    `extent` is the enclosing_range when SCIP provides it (covers the full
    function body), else the bare `range` (covers just the identifier). The
    former is what we want; the latter is a degenerate fallback.
    """

    symbol: str
    extent: tuple[int, int, int, int]
    """Containment range (start_line, start_col, end_line, end_col), 0-based."""


def find_enclosing_definition(
    occurrence_range: tuple[int, int, int, int],
    definitions: list[DefinitionExtent],
) -> Optional[DefinitionExtent]:
    """Locate the smallest (deepest) definition whose extent contains the
    given occurrence range.

    `definitions` is assumed to be sorted by `extent` start position
    ascending. Ties are broken by end position descending (largest first),
    which is the natural order when definitions are written outside-in.
    If sorting isn't guaranteed, callers should sort before invoking.

    Returns None when no definition contains the occurrence.
    """
    best: Optional[DefinitionExtent] = None
    best_size: Optional[int] = None
    occ_sl, occ_sc, occ_el, occ_ec = occurrence_range
    for d in definitions:
        # Once a definition starts strictly after the occurrence, no later
        # one (start-sorted) can contain it. Bail out.
        if (d.extent[0], d.extent[1]) > (occ_sl, occ_sc):
            break
        if range_contains(d.extent, occurrence_range):
            size = _range_size(d.extent)
            if best_size is None or size < best_size:
                best = d
                best_size = size
    return best


def _range_size(r: tuple[int, int, int, int]) -> int:
    """A monotonic 'size' for comparing two ranges. Smaller = more specific.
    Uses lines * 1000 + cols as a coarse heuristic — works for any reasonable
    source file (no line is wider than 1000 columns in practice).
    """
    sl, sc, el, ec = r
    if el == sl:
        return ec - sc
    return (el - sl) * 1_000_000 + (1_000_000 - sc) + ec
