"""End-to-end smoke test for ScipClangBackend using a synthesized .scip file.

The real backend invokes the `scip-clang` binary as a subprocess to produce
a `.scip` Index. To exercise the parsing + assembly path without needing
the actual binary, we construct a minimal protobuf `Index` in memory,
serialize it to a temp file, and feed it to the backend's internal parser.

This covers everything except the subprocess invocation itself (a
straightforward `subprocess.run` call, separately validated in Phase 6 with
a real scip-clang install).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


def _ensure_scip_pb2():
    """Make `from hmopt.indexing._generated import scip_pb2` importable in
    a test process that doesn't have the rest of `hmopt.indexing` runnable
    (which transitively requires llama_index, neo4j, etc.).
    """
    repo_root = Path(__file__).resolve().parent.parent
    gen_dir = repo_root / "src" / "hmopt" / "indexing" / "_generated"
    pb2_path = gen_dir / "scip_pb2.py"
    if not pb2_path.exists():
        pytest.skip(
            "scip_pb2.py not generated yet; run `bash scripts/gen_scip_pb2.sh`"
        )
    # Create a synthetic top-level module path that lets `_scip_translation`
    # and `scip_clang.py` resolve their `from .._generated import scip_pb2`.
    # We do this by loading the package shell manually.
    if "hmopt_test_indexing" not in sys.modules:
        import types

        pkg = types.ModuleType("hmopt_test_indexing")
        pkg.__path__ = [str(repo_root / "src" / "hmopt" / "indexing")]
        sys.modules["hmopt_test_indexing"] = pkg

        gen_pkg = types.ModuleType("hmopt_test_indexing._generated")
        gen_pkg.__path__ = [str(gen_dir)]
        sys.modules["hmopt_test_indexing._generated"] = gen_pkg

        spec = importlib.util.spec_from_file_location(
            "hmopt_test_indexing._generated.scip_pb2", pb2_path
        )
        assert spec is not None and spec.loader is not None
        pb2 = importlib.util.module_from_spec(spec)
        sys.modules["hmopt_test_indexing._generated.scip_pb2"] = pb2
        spec.loader.exec_module(pb2)
        return pb2

    return sys.modules["hmopt_test_indexing._generated.scip_pb2"]


@pytest.fixture(scope="module")
def scip_pb2():
    """Provide the protobuf bindings module to the tests."""
    try:
        import google.protobuf  # noqa: F401
    except ImportError:
        pytest.skip("google.protobuf not installed (pip install protobuf)")
    return _ensure_scip_pb2()


def _load_backend_module():
    """Load `scip_clang.py` directly, wiring its `_generated.scip_pb2` import
    against the module loaded by `_ensure_scip_pb2`. This sidesteps the full
    `hmopt.indexing.__init__` chain which requires runtime deps unrelated to
    SCIP parsing.
    """
    _ensure_scip_pb2()
    repo_root = Path(__file__).resolve().parent.parent
    backends_dir = repo_root / "src" / "hmopt" / "indexing" / "backends"
    models_path = repo_root / "src" / "hmopt" / "indexing" / "models.py"
    tx_path = backends_dir / "_scip_translation.py"
    backend_path = backends_dir / "scip_clang.py"

    # Set up a minimal synthetic `hmopt.indexing` namespace.
    import types

    if "hmopt_test_indexing.models" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "hmopt_test_indexing.models", models_path
        )
        models = importlib.util.module_from_spec(spec)
        sys.modules["hmopt_test_indexing.models"] = models
        spec.loader.exec_module(models)

    if "hmopt_test_indexing.backends" not in sys.modules:
        pkg = types.ModuleType("hmopt_test_indexing.backends")
        pkg.__path__ = [str(backends_dir)]
        sys.modules["hmopt_test_indexing.backends"] = pkg

    if "hmopt_test_indexing.backends._scip_translation" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "hmopt_test_indexing.backends._scip_translation", tx_path
        )
        tx = importlib.util.module_from_spec(spec)
        sys.modules["hmopt_test_indexing.backends._scip_translation"] = tx
        spec.loader.exec_module(tx)

    if "hmopt_test_indexing.backends.scip_clang" not in sys.modules:
        # Rewrite the relative imports inside scip_clang.py to point at our
        # synthetic namespace. Easiest: pre-load the source into a string,
        # patch the imports, and exec it.
        src = backend_path.read_text(encoding="utf-8")
        src = src.replace(
            "from ..models import", "from hmopt_test_indexing.models import"
        )
        src = src.replace(
            "from . import _scip_translation as tx",
            "from hmopt_test_indexing.backends import _scip_translation as tx",
        )
        # The source now uses an absolute submodule import to avoid the
        # "partially initialized module" error in real `pip install -e .`
        # environments under Python 3.11+. Rewrite it to the synthetic
        # namespace this fixture sets up.
        src = src.replace(
            "import hmopt.indexing.backends._scip_translation as tx",
            "import hmopt_test_indexing.backends._scip_translation as tx",
        )
        src = src.replace(
            "from .._generated import scip_pb2",
            "from hmopt_test_indexing._generated import scip_pb2",
        )
        backend_module = types.ModuleType("hmopt_test_indexing.backends.scip_clang")
        backend_module.__file__ = str(backend_path)
        backend_module.__package__ = "hmopt_test_indexing.backends"
        # Register in sys.modules BEFORE exec so dataclass decorator's
        # annotation resolution (which looks up `sys.modules[cls.__module__]`)
        # can find this module while the class bodies are being defined.
        sys.modules["hmopt_test_indexing.backends.scip_clang"] = backend_module
        exec(compile(src, str(backend_path), "exec"), backend_module.__dict__)

    return sys.modules["hmopt_test_indexing.backends.scip_clang"]


@pytest.fixture(scope="module")
def backend_mod(scip_pb2):
    return _load_backend_module()


# ---------------------------------------------------------------------------
# Index construction helper
# ---------------------------------------------------------------------------


def _build_minimal_index(scip_pb2):
    """Build a tiny SCIP Index representing this C source:

        // foo.c, 0-based lines as shown
        // 0: void helper(int x) {           <- definition at lines 0-2
        // 1:   return;
        // 2: }
        // 3:
        // 4: int caller(void) {              <- definition at lines 4-7
        // 5:   helper(42);                   <- call site at line 5, cols 2-8
        // 6:   return 0;
        // 7: }

    `helper` and `caller` are both defined in this Document. The call from
    `caller` to `helper` exercises:
      - call_site_path / call_site_line filling
      - enclosing-function discovery (the ref at line 5 lives inside caller's
        enclosing_range)
      - cross-symbol relation building (src=caller, dst=helper)
    """
    idx = scip_pb2.Index()
    doc = idx.documents.add()
    doc.relative_path = "foo.c"
    doc.language = "C"

    # SymbolInformation
    info_helper = doc.symbols.add()
    info_helper.symbol = "cxx . . . helper()."
    info_helper.kind = scip_pb2.SymbolInformation.Function
    info_helper.display_name = "helper"

    info_caller = doc.symbols.add()
    info_caller.symbol = "cxx . . . caller()."
    info_caller.kind = scip_pb2.SymbolInformation.Function
    info_caller.display_name = "caller"

    # Occurrence 1: definition of helper at line 0, with enclosing range 0-2
    occ = doc.occurrences.add()
    occ.range.extend([0, 5, 0, 11])  # identifier 'helper' on line 0
    occ.enclosing_range.extend([0, 0, 2, 1])  # full function body 0..2
    occ.symbol = "cxx . . . helper()."
    occ.symbol_roles = scip_pb2.Definition
    occ.syntax_kind = scip_pb2.IdentifierFunctionDefinition

    # Occurrence 2: definition of caller at line 4, enclosing range 4-7
    occ = doc.occurrences.add()
    occ.range.extend([4, 4, 4, 10])
    occ.enclosing_range.extend([4, 0, 7, 1])
    occ.symbol = "cxx . . . caller()."
    occ.symbol_roles = scip_pb2.Definition
    occ.syntax_kind = scip_pb2.IdentifierFunctionDefinition

    # Occurrence 3: reference to helper from inside caller's body, line 5
    occ = doc.occurrences.add()
    occ.range.extend([5, 2, 5, 8])  # 'helper' on line 5, cols 2..8
    occ.symbol = "cxx . . . helper()."
    occ.symbol_roles = 0  # plain reference, no Definition bit
    occ.syntax_kind = scip_pb2.IdentifierFunction

    return idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_yields_chunks_and_relations(tmp_path, scip_pb2, backend_mod):
    """The parser must emit one CodeChunk per definition and one CodeRelation
    per reference, with backend_origin tagged and call_site_line filled."""
    idx = _build_minimal_index(scip_pb2)
    scip_file = tmp_path / "index.scip"
    scip_file.write_bytes(idx.SerializeToString())

    # Synthesize source file so chunk text extraction has data to work with.
    source = (
        "void helper(int x) {\n"
        "  return;\n"
        "}\n"
        "\n"
        "int caller(void) {\n"
        "  helper(42);\n"
        "  return 0;\n"
        "}\n"
    )
    (tmp_path / "foo.c").write_text(source, encoding="utf-8")

    backend = backend_mod.ScipClangBackend()
    diagnostics: dict = {}
    code_index = backend._parse_scip(scip_file, tmp_path, diagnostics)

    # Two definitions → two chunks
    assert len(code_index.chunks) == 2
    by_name = {c.symbol_name: c for c in code_index.chunks}
    assert "helper" in by_name and "caller" in by_name
    helper_chunk = by_name["helper"]
    caller_chunk = by_name["caller"]

    # Chunk fields are correctly tagged for the scip-clang backend
    assert helper_chunk.backend_origin == "scip-clang"
    assert helper_chunk.scip_symbol == "cxx . . . helper()."
    assert helper_chunk.kind == "function"
    assert helper_chunk.parser == "scip-clang"
    assert helper_chunk.start_line == 1  # 0-based 0 → 1-based 1
    assert helper_chunk.end_line == 3  # enclosing_range end 2 → 1-based 3

    assert caller_chunk.start_line == 5
    assert caller_chunk.end_line == 8

    # Chunk text should be the body lines (extracted from disk).
    assert "helper" in helper_chunk.text
    assert "caller" in caller_chunk.text

    # One reference → one relation: caller -> helper (calls)
    assert len(code_index.relations) == 1
    rel = code_index.relations[0]
    assert rel.kind == "calls"
    assert rel.backend_origin == "scip-clang"
    assert rel.src_name == "caller"
    assert rel.dst_name == "helper"
    assert rel.call_site_path == "foo.c"
    assert rel.call_site_line == 6  # 0-based line 5 → 1-based 6
    assert rel.call_site_col == 3  # 0-based col 2 → 1-based 3
    assert rel.syntax_kind == scip_pb2.IdentifierFunction


def test_parse_external_reference_is_marked_external(tmp_path, scip_pb2, backend_mod):
    """A reference to a symbol that has no definition occurrence anywhere in
    the index should yield a relation whose dst_id starts with `external::`."""
    idx = scip_pb2.Index()
    doc = idx.documents.add()
    doc.relative_path = "foo.c"

    # One internal definition + one reference to an unknown external symbol
    info_caller = doc.symbols.add()
    info_caller.symbol = "cxx . . . caller()."
    info_caller.kind = scip_pb2.SymbolInformation.Function

    occ_def = doc.occurrences.add()
    occ_def.range.extend([0, 0, 0, 6])
    occ_def.enclosing_range.extend([0, 0, 4, 1])
    occ_def.symbol = "cxx . . . caller()."
    occ_def.symbol_roles = scip_pb2.Definition
    occ_def.syntax_kind = scip_pb2.IdentifierFunctionDefinition

    occ_ref = doc.occurrences.add()
    occ_ref.range.extend([2, 4, 2, 10])
    occ_ref.symbol = "cxx . . . printf()."  # not defined anywhere in this index
    occ_ref.symbol_roles = 0
    occ_ref.syntax_kind = scip_pb2.IdentifierFunction

    scip_file = tmp_path / "index.scip"
    scip_file.write_bytes(idx.SerializeToString())
    (tmp_path / "foo.c").write_text(
        "int caller(void) {\n  printf(\"hi\");\n  return 0;\n}\n",
        encoding="utf-8",
    )

    backend = backend_mod.ScipClangBackend()
    code_index = backend._parse_scip(scip_file, tmp_path, {})

    assert len(code_index.relations) == 1
    rel = code_index.relations[0]
    assert rel.dst_id.startswith("external::"), rel.dst_id
    assert rel.dst_id == "external::cxx . . . printf()."
    assert rel.dst_path is None
    assert rel.call_site_line == 3


def test_parse_skips_local_and_punctuation_occurrences(
    tmp_path, scip_pb2, backend_mod
):
    """Local-scheme occurrences and pure-punctuation syntax kinds yield no
    chunks / relations."""
    idx = scip_pb2.Index()
    doc = idx.documents.add()
    doc.relative_path = "foo.c"

    # Local-scheme definition — should be skipped (not a real exported symbol).
    occ_local = doc.occurrences.add()
    occ_local.range.extend([0, 0, 0, 5])
    occ_local.symbol = "local 1"
    occ_local.symbol_roles = scip_pb2.Definition

    # Punctuation occurrence — no edge meaning, must be skipped.
    occ_punct = doc.occurrences.add()
    occ_punct.range.extend([1, 0, 1, 1])
    occ_punct.symbol = "cxx . . . somewhere()."
    occ_punct.symbol_roles = 0
    occ_punct.syntax_kind = scip_pb2.PunctuationDelimiter

    scip_file = tmp_path / "index.scip"
    scip_file.write_bytes(idx.SerializeToString())

    backend = backend_mod.ScipClangBackend()
    code_index = backend._parse_scip(scip_file, tmp_path, {})

    assert code_index.chunks == []
    assert code_index.relations == []


# ---------------------------------------------------------------------------
# Regression: descriptor-suffix fallback recovers relations when
# scip-clang leaves syntax_kind=UnspecifiedSyntaxKind(0).
#
# scip-clang 0.3.1 + BiSheng (HarmonyOS kernel toolchain) emits the
# overwhelming majority of reference occurrences with syntax_kind=0
# (47,680 occurrences in a sysmgr build, of which ~80% are references
# but only ~5% of those land in the syntax_kind table). Without the
# fallback, the indexer drops 30k+ references and Neo4j ends up with
# zero scip-clang relation edges.
# ---------------------------------------------------------------------------


def test_parse_recovers_relation_when_syntax_kind_unspecified(
    tmp_path, scip_pb2, backend_mod
):
    """The reference occurrence carries syntax_kind=0 (default value, scip-clang
    didn't set it), but the SCIP symbol descriptor `helper().` still uniquely
    identifies a method call. The backend MUST recover the relation kind
    from the descriptor's `method` suffix and emit a `:calls` edge."""
    idx = scip_pb2.Index()
    doc = idx.documents.add()
    doc.relative_path = "foo.c"
    doc.language = "C"

    for name in ("helper", "caller"):
        info = doc.symbols.add()
        info.symbol = f"cxx . . . {name}()."
        info.kind = scip_pb2.SymbolInformation.Function
        info.display_name = name

    # helper definition (lines 0..2)
    occ = doc.occurrences.add()
    occ.range.extend([0, 5, 0, 11])
    occ.enclosing_range.extend([0, 0, 2, 1])
    occ.symbol = "cxx . . . helper()."
    occ.symbol_roles = scip_pb2.Definition
    occ.syntax_kind = scip_pb2.IdentifierFunctionDefinition

    # caller definition (lines 4..7)
    occ = doc.occurrences.add()
    occ.range.extend([4, 4, 4, 10])
    occ.enclosing_range.extend([4, 0, 7, 1])
    occ.symbol = "cxx . . . caller()."
    occ.symbol_roles = scip_pb2.Definition
    occ.syntax_kind = scip_pb2.IdentifierFunctionDefinition

    # *** The critical occurrence: reference to helper from inside caller's
    # *** body, with syntax_kind LEFT AT ITS DEFAULT (UnspecifiedSyntaxKind=0).
    occ = doc.occurrences.add()
    occ.range.extend([5, 2, 5, 8])
    occ.symbol = "cxx . . . helper()."
    occ.symbol_roles = 0
    # NOTE: occ.syntax_kind intentionally not set — protobuf default = 0.

    scip_file = tmp_path / "index.scip"
    scip_file.write_bytes(idx.SerializeToString())
    (tmp_path / "foo.c").write_text(
        "void helper(int x) {\n"
        "  return;\n"
        "}\n"
        "\n"
        "int caller(void) {\n"
        "  helper(42);\n"
        "  return 0;\n"
        "}\n",
        encoding="utf-8",
    )

    backend = backend_mod.ScipClangBackend()
    code_index = backend._parse_scip(scip_file, tmp_path, {})

    assert len(code_index.relations) == 1, (
        "fallback inferred via descriptor suffix should recover one :calls edge"
    )
    rel = code_index.relations[0]
    assert rel.kind == "calls"
    assert rel.backend_origin == "scip-clang"
    assert rel.src_name == "caller"
    assert rel.dst_name == "helper"
    assert rel.call_site_line == 6
    # The raw syntax_kind on the relation reflects what scip-clang emitted,
    # which was 0 — we recovered the relation kind from the descriptor,
    # not by patching this field.
    assert rel.syntax_kind == 0


def test_parse_does_not_recover_when_syntax_kind_is_punctuation(
    tmp_path, scip_pb2, backend_mod
):
    """Tightening the fallback: when scip-clang DOES set syntax_kind but it
    maps to None in our table (e.g. PunctuationDelimiter=2, Comment=1), we
    must respect that signal and skip the occurrence. Otherwise an incidental
    method-shaped token in punctuation context would manufacture a phantom
    :calls edge."""
    idx = scip_pb2.Index()
    doc = idx.documents.add()
    doc.relative_path = "x.c"
    doc.language = "C"

    # A "caller" definition so any spurious recovered edge would actually
    # find a src_def via the enclosing_range and become a relation.
    occ = doc.occurrences.add()
    occ.range.extend([0, 4, 0, 10])
    occ.enclosing_range.extend([0, 0, 4, 1])
    occ.symbol = "cxx . . . caller()."
    occ.symbol_roles = scip_pb2.Definition
    occ.syntax_kind = scip_pb2.IdentifierFunctionDefinition
    info = doc.symbols.add()
    info.symbol = "cxx . . . caller()."
    info.kind = scip_pb2.SymbolInformation.Function

    # Punctuation occurrence that happens to carry a method-shaped symbol.
    # syntax_kind != 0, so the fallback MUST NOT engage.
    occ = doc.occurrences.add()
    occ.range.extend([1, 0, 1, 1])
    occ.symbol = "cxx . . . somewhere()."
    occ.symbol_roles = 0
    occ.syntax_kind = scip_pb2.PunctuationDelimiter

    scip_file = tmp_path / "index.scip"
    scip_file.write_bytes(idx.SerializeToString())

    backend = backend_mod.ScipClangBackend()
    code_index = backend._parse_scip(scip_file, tmp_path, {})

    assert code_index.relations == [], (
        "explicit non-edge syntax_kind values must be respected — fallback "
        "should only engage for UnspecifiedSyntaxKind(0)"
    )


# ---------------------------------------------------------------------------
# Regression: import form must stay absolute
# ---------------------------------------------------------------------------


def test_scip_clang_uses_absolute_translation_import():
    """The `_scip_translation` helper module MUST be imported by absolute
    path, NOT via `from . import _scip_translation as tx`.

    When `hmopt.indexing.backends.__init__` re-exports `ScipClangBackend`,
    `scip_clang.py` is loaded while the `backends` package is still
    partially initialized. Under Python 3.11+ the relative `from . import`
    form raises:

        ImportError: cannot import name '_scip_translation' from
        partially initialized module 'hmopt.indexing.backends'

    The absolute `import hmopt.indexing.backends._scip_translation as tx`
    form goes through sys.modules directly and is safe even mid-init.

    The other unit tests in this file string-replace the import line into
    a synthetic namespace before exec'ing the module, so they CANNOT
    catch a regression that puts the relative form back. This test
    inspects the source text directly as a lightweight guard.
    """
    src_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "hmopt" / "indexing" / "backends" / "scip_clang.py"
    )
    src = src_path.read_text(encoding="utf-8")
    assert "from . import _scip_translation as tx" not in src, (
        "scip_clang.py must NOT use `from . import _scip_translation as tx` "
        "— that form fails under Python 3.11+ when backends/__init__.py "
        "imports scip_clang while the package is mid-init. Use the absolute "
        "submodule import form (see comment in scip_clang.py)."
    )
    assert "import hmopt.indexing.backends._scip_translation as tx" in src
