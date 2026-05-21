"""Unit tests for hmopt.indexing.backends._scip_translation.

Covers SCIP symbol-string parsing, syntax-kind / symbol-kind mapping, range
normalization, and enclosing-definition lookup.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load _scip_translation directly to avoid pulling the full hmopt.indexing
# package init (which transitively requires llama_index, neo4j, etc.).
_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "hmopt"
    / "indexing"
    / "backends"
    / "_scip_translation.py"
)


def _load_translation_module():
    spec = importlib.util.spec_from_file_location(
        "_scip_translation_under_test", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_scip_translation_under_test"] = module
    spec.loader.exec_module(module)
    return module


tx = _load_translation_module()


# ---------------------------------------------------------------------------
# parse_scip_symbol
# ---------------------------------------------------------------------------


def test_parse_simple_c_function():
    """A free C function — no namespace, no file in descriptor.

    scip-clang puts the file in Document.relative_path, NOT in the symbol's
    descriptor path. The descriptor is just the function name plus method
    suffix.
    """
    parsed = tx.parse_scip_symbol("cxx . . . bar().")
    assert parsed is not None
    assert parsed.scheme == "cxx"
    assert parsed.manager == "."
    assert parsed.package_name == "."
    assert parsed.version == "."
    assert [d.name for d in parsed.descriptors] == ["bar"]
    assert [d.suffix for d in parsed.descriptors] == ["method"]
    assert parsed.qualname == "bar"
    assert parsed.kind == "function"


def test_parse_struct_field():
    """A struct field — type descriptor + term descriptor."""
    parsed = tx.parse_scip_symbol("cxx . . . Outer#field.")
    assert parsed is not None
    assert parsed.qualname == "Outer::field"
    # A `.` (term) terminator with a struct-member meaning is mapped to
    # "variable" in our kind vocabulary; the existing clangd backend does
    # the same for C struct fields.
    assert parsed.kind == "variable"


def test_parse_struct_type_only():
    """A struct type symbol — descriptor ends in '#'."""
    parsed = tx.parse_scip_symbol("cxx . . . Outer#")
    assert parsed is not None
    assert parsed.qualname == "Outer"
    assert parsed.kind == "type"


def test_parse_macro():
    """A C preprocessor macro — descriptor ends in '!'."""
    parsed = tx.parse_scip_symbol("cxx . . . MY_MACRO!")
    assert parsed is not None
    assert parsed.qualname == "MY_MACRO"
    assert parsed.kind == "macro"


def test_parse_namespaced_function():
    """C++ namespaced function — namespace descriptors ARE part of qualname.

    Per SCIP grammar, names containing `.`, `/`, etc. must be backtick-escaped,
    so an unescaped namespace in the descriptor is always a real C++ namespace
    (not a directory or filename). Hence we include it in qualname.
    """
    parsed = tx.parse_scip_symbol("cxx . . . ns1/ns2/bar().")
    assert parsed is not None
    assert parsed.qualname == "ns1::ns2::bar"
    assert parsed.kind == "function"


def test_parse_method_with_disambiguator():
    """Method with non-empty disambiguator (overload index, etc.)."""
    parsed = tx.parse_scip_symbol("cxx . . . Cls#overloaded(int).")
    assert parsed is not None
    method_desc = parsed.descriptors[-1]
    assert method_desc.name == "overloaded"
    assert method_desc.suffix == "method"
    assert method_desc.disambiguator == "int"
    assert parsed.qualname == "Cls::overloaded"


def test_parse_with_backtick_escaped_filename_as_namespace():
    """Some SCIP emitters (not standard scip-clang) embed the file as a
    backtick-escaped namespace. Make sure we parse it correctly even though
    the filename includes a `.`."""
    parsed = tx.parse_scip_symbol("cxx . . . `foo.c`/bar().")
    assert parsed is not None
    assert [d.name for d in parsed.descriptors] == ["foo.c", "bar"]
    assert parsed.descriptors[0].suffix == "namespace"
    assert parsed.descriptors[1].suffix == "method"
    assert parsed.qualname == "foo.c::bar"


def test_parse_local_symbol():
    """Local-scheme symbols (function-local variables) get a degenerate parse."""
    parsed = tx.parse_scip_symbol("local 42")
    assert parsed is not None
    assert parsed.is_local
    assert parsed.scheme == "local"
    assert parsed.qualname == "42"


def test_parse_empty_returns_none():
    assert tx.parse_scip_symbol("") is None
    assert tx.parse_scip_symbol("   ") is None


def test_parse_malformed_returns_none():
    # Missing descriptor path.
    assert tx.parse_scip_symbol("cxx . . .") is None
    # Method without closing paren.
    assert tx.parse_scip_symbol("cxx . . . foo.c/bar(") is None


def test_parse_backtick_escaped_name():
    """Names with internal spaces use backtick escaping; `` is an escaped tick."""
    parsed = tx.parse_scip_symbol("cxx . . . file.c/`weird name`#")
    assert parsed is not None
    assert parsed.descriptors[-1].name == "weird name"
    assert parsed.descriptors[-1].suffix == "type"
    # Doubled backtick → literal `
    parsed2 = tx.parse_scip_symbol("cxx . . . file.c/`tick``inside`#")
    assert parsed2 is not None
    assert parsed2.descriptors[-1].name == "tick`inside"


# ---------------------------------------------------------------------------
# scip_syntax_kind_to_relation_kind
# ---------------------------------------------------------------------------


def test_syntax_kind_mapping_for_calls():
    assert (
        tx.scip_syntax_kind_to_relation_kind(tx.SYNTAX_KIND_IDENTIFIER_FUNCTION)
        == "calls"
    )
    assert (
        tx.scip_syntax_kind_to_relation_kind(
            tx.SYNTAX_KIND_IDENTIFIER_FUNCTION_DEFINITION
        )
        == "calls"
    )


def test_syntax_kind_mapping_for_types_and_macros():
    assert (
        tx.scip_syntax_kind_to_relation_kind(tx.SYNTAX_KIND_IDENTIFIER_TYPE)
        == "uses_type"
    )
    assert (
        tx.scip_syntax_kind_to_relation_kind(tx.SYNTAX_KIND_IDENTIFIER_BUILTIN_TYPE)
        == "uses_type"
    )
    assert (
        tx.scip_syntax_kind_to_relation_kind(tx.SYNTAX_KIND_IDENTIFIER_MACRO)
        == "uses_macro"
    )


def test_syntax_kind_punctuation_returns_none():
    """Comments / punctuation / keywords have no relation meaning."""
    # SCIP enum values for Comment=1, PunctuationDelimiter=2, Keyword=4.
    assert tx.scip_syntax_kind_to_relation_kind(1) is None
    assert tx.scip_syntax_kind_to_relation_kind(2) is None
    assert tx.scip_syntax_kind_to_relation_kind(4) is None


# ---------------------------------------------------------------------------
# infer_relation_kind_from_descriptor — fallback for when scip-clang
# leaves syntax_kind=UnspecifiedSyntaxKind(0) on reference occurrences.
# This was the root cause of ~80% of refs being silently dropped on
# scip-clang 0.3.1 + BiSheng-generated compile_commands.
# ---------------------------------------------------------------------------


def test_infer_relation_kind_from_method_descriptor():
    """`name().` ending → calls. Most function calls in C/C++ scip output."""
    parsed = tx.parse_scip_symbol("cxx . . . foo().")
    assert parsed is not None
    assert tx.infer_relation_kind_from_descriptor(parsed) == "calls"


def test_infer_relation_kind_from_namespaced_method_descriptor():
    """Last descriptor's suffix wins; namespace prefix doesn't change kind."""
    parsed = tx.parse_scip_symbol("cxx . . . MyClass#bar().")
    assert parsed is not None
    assert parsed.descriptors[-1].suffix == "method"
    assert tx.infer_relation_kind_from_descriptor(parsed) == "calls"


def test_infer_relation_kind_from_macro_descriptor():
    """`name!` ending → uses_macro."""
    parsed = tx.parse_scip_symbol("cxx . . . LOG_INFO!")
    assert parsed is not None
    assert tx.infer_relation_kind_from_descriptor(parsed) == "uses_macro"


def test_infer_relation_kind_from_type_descriptor():
    """`name#` ending → uses_type. Covers struct/union/enum/typedef refs."""
    parsed = tx.parse_scip_symbol("cxx . . . task_struct#")
    assert parsed is not None
    assert tx.infer_relation_kind_from_descriptor(parsed) == "uses_type"


def test_infer_relation_kind_from_type_parameter_descriptor():
    parsed = tx.parse_scip_symbol("cxx . . . [T]")
    assert parsed is not None
    assert parsed.descriptors[-1].suffix == "type_parameter"
    assert tx.infer_relation_kind_from_descriptor(parsed) == "uses_type"


def test_infer_relation_kind_from_term_descriptor_falls_back_to_references():
    """`name.` (bare term — variable, field, parameter) maps to the generic
    `references` edge. We deliberately do NOT classify these as 'calls'
    because we can't distinguish a function-pointer reference from a plain
    variable read without more context."""
    parsed = tx.parse_scip_symbol("cxx . . . jiffies.")
    assert parsed is not None
    assert tx.infer_relation_kind_from_descriptor(parsed) == "references"


def test_infer_relation_kind_from_empty_descriptors_returns_none():
    """A ParsedSymbol with no descriptors carries no symbolic identity —
    the caller should skip the occurrence entirely."""
    parsed = tx.parse_scip_symbol("cxx . . . ")
    # Implementation choice: parse_scip_symbol returns None on empty
    # descriptor path; we additionally guard infer_* in case the caller
    # constructs ParsedSymbol manually.
    assert parsed is None or tx.infer_relation_kind_from_descriptor(parsed) is None


# ---------------------------------------------------------------------------
# scip_symbol_kind_to_node_kind
# ---------------------------------------------------------------------------


def test_symbol_kind_mapping_for_common_c_kinds():
    assert tx.scip_symbol_kind_to_node_kind(tx.SYMBOL_KIND_FUNCTION) == "function"
    assert tx.scip_symbol_kind_to_node_kind(tx.SYMBOL_KIND_STRUCT) == "struct"
    assert tx.scip_symbol_kind_to_node_kind(tx.SYMBOL_KIND_ENUM) == "enum"
    assert tx.scip_symbol_kind_to_node_kind(tx.SYMBOL_KIND_MACRO) == "macro"
    assert tx.scip_symbol_kind_to_node_kind(tx.SYMBOL_KIND_FIELD) == "field"
    assert tx.scip_symbol_kind_to_node_kind(tx.SYMBOL_KIND_VARIABLE) == "variable"


def test_symbol_kind_unspecified_returns_unknown():
    """SymbolInformation.Kind == 0 (UnspecifiedKind) → 'unknown'."""
    assert tx.scip_symbol_kind_to_node_kind(0) == "unknown"


def test_symbol_kind_union_maps_to_struct():
    """C unions share Neo4j label with structs in the existing schema."""
    assert tx.scip_symbol_kind_to_node_kind(tx.SYMBOL_KIND_UNION) == "struct"


# ---------------------------------------------------------------------------
# normalize_range
# ---------------------------------------------------------------------------


def test_normalize_range_4_ints_multiline():
    assert tx.normalize_range([10, 5, 12, 8]) == (10, 5, 12, 8)


def test_normalize_range_3_ints_single_line():
    """A 3-int range is same-line: [start_line, start_col, end_col]."""
    assert tx.normalize_range([10, 5, 12]) == (10, 5, 10, 12)


def test_normalize_range_malformed_returns_none():
    assert tx.normalize_range([]) is None
    assert tx.normalize_range([1, 2]) is None
    assert tx.normalize_range([1, 2, 3, 4, 5]) is None


def test_range_contains_basic():
    outer = (0, 0, 100, 0)
    inner = (10, 5, 10, 8)
    assert tx.range_contains(outer, inner)
    assert not tx.range_contains(inner, outer)


def test_range_contains_inclusive_boundary():
    """A range contains itself (inclusive bounds)."""
    r = (10, 0, 20, 5)
    assert tx.range_contains(r, r)


def test_range_contains_partial_overlap_is_not_containment():
    a = (10, 0, 20, 0)
    b = (15, 0, 25, 0)
    assert not tx.range_contains(a, b)
    assert not tx.range_contains(b, a)


# ---------------------------------------------------------------------------
# find_enclosing_definition
# ---------------------------------------------------------------------------


def _make_defs(*specs: tuple[str, tuple[int, int, int, int]]):
    """Build a list[DefinitionExtent] from (symbol, extent) pairs."""
    return [tx.DefinitionExtent(symbol=s, extent=e) for s, e in specs]


def test_enclosing_definition_basic_function_body():
    """Reference at line 12 is inside function body lines 10-20."""
    defs = _make_defs(
        ("cxx . . . file.c/foo().", (10, 0, 20, 1)),
    )
    occ = (12, 4, 12, 7)
    result = tx.find_enclosing_definition(occ, defs)
    assert result is not None
    assert result.symbol == "cxx . . . file.c/foo()."


def test_enclosing_definition_picks_innermost_nested():
    """Inner function body is inside outer translation unit's range — pick inner."""
    defs = _make_defs(
        ("cxx . . . file.c/outer().", (0, 0, 100, 0)),
        ("cxx . . . file.c/inner().", (40, 0, 60, 1)),
    )
    occ = (50, 4, 50, 10)
    result = tx.find_enclosing_definition(occ, defs)
    assert result is not None
    assert result.symbol == "cxx . . . file.c/inner()."


def test_enclosing_definition_returns_none_when_no_match():
    defs = _make_defs(
        ("cxx . . . file.c/foo().", (10, 0, 20, 1)),
    )
    occ = (50, 0, 50, 5)
    assert tx.find_enclosing_definition(occ, defs) is None


def test_enclosing_definition_handles_unsorted_input_gracefully():
    """If callers pass unsorted defs the function still returns SOMETHING
    correct (just may have to scan more). We don't promise correctness when
    unsorted, but we do verify it doesn't crash."""
    defs = _make_defs(
        ("cxx . . . file.c/inner().", (40, 0, 60, 1)),
        ("cxx . . . file.c/outer().", (0, 0, 100, 0)),
    )
    occ = (50, 4, 50, 10)
    # The function bails on the first def whose start > occurrence start;
    # since defs[0].start = (40,0) <= (50,4), it gets considered; defs[1].start
    # = (0,0) <= (50,4) too. Both contain the occurrence. The smaller wins.
    # NB: we rely on size comparison, not iteration order, for "innermost".
    result = tx.find_enclosing_definition(occ, defs)
    assert result is not None
    assert result.symbol == "cxx . . . file.c/inner()."
