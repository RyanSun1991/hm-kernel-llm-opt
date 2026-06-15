"""Dual-tree parity tests (review P0-3).

The hub (`hm-skill-hub/tools/`) and the consumer (`src/hmopt/`) deliberately
carry duplicated logic so the hub stays split-out-ready. That is only safe if
every duplicated surface is pinned by a parity test — otherwise it drifts (the
`polarity` "do: avoid X" bug shipped on one side only and was caught here). This
file pins the surfaces the session review flagged: polarity, the token-hashing
embedder, tokenization, and the common-path strength ordering.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "hm-skill-hub" / "tools"))

# hub side
import similarity as hub_sim  # type: ignore
from hub_records import HubRecord  # type: ignore

# consumer side
from hmopt.memory.local_curator import CuratorItem
from hmopt.skillhub.embeddings import HashingEmbedder
from hmopt.skillhub.embeddings import tokenize as consumer_tokenize


def _hub_item(do="", delta=None, status="active"):
    fields = {"do_or_dont": do, "status": status}
    if delta is not None:
        fields["delta_pct"] = delta
    return HubRecord(id="X", schema="memory_item", path="x", fields=fields, body="")


def _consumer_item(do="", delta=None, status="active"):
    return CuratorItem(id="X", do_or_dont=do, delta_pct=delta, status=status)


# ---- polarity parity (the surface that already drifted) --------------------

POLARITY_CASES = [
    ("do: avoid spending an iteration on cold paths", None, "active", -1),
    ("do: hoist the loop-invariant read", None, "active", 1),
    ("don't blanket-inline kworker entries", None, "active", -1),
    ("watch out for single-image tests", None, "active", -1),
    ("", -0.8, "active", 1),    # negative delta == improvement
    ("", 0.6, "active", -1),
    ("", None, "landed", 1),
    ("", None, "rejected", -1),
    ("", None, "active", 0),
]


def test_polarity_parity_hub_vs_consumer():
    for do, delta, status, expected in POLARITY_CASES:
        h = _hub_item(do, delta, status).polarity()
        c = _consumer_item(do, delta, status).polarity()
        assert h == c == expected, f"polarity drift on ({do!r},{delta},{status}): hub={h} consumer={c}"


# ---- embedder + tokenizer byte-equality ------------------------------------

EMBED_STRINGS = [
    "mm/vmscan.c::shrink_node sc->priority hoist",
    "hoist the loop-invariant out of the reclaim loop",
    "batch-coalesce round-trip workqueue",
    "",
]


def test_tokenize_parity():
    for s in EMBED_STRINGS:
        assert hub_sim.tokenize(s) == consumer_tokenize(s), f"tokenize drift on {s!r}"


def test_embedder_byte_equality():
    e = HashingEmbedder(dim=256)  # hub similarity.embed defaults to dim=256
    for s in EMBED_STRINGS:
        assert hub_sim.embed(s) == e.embed(s), f"embedder drift on {s!r}"


# ---- strength parity on the shared (maturity, confirmations) path ----------

def test_strength_parity_on_maturity_path():
    # Both sides must agree when a record carries an explicit maturity (the common
    # memory_item case). (Confidence-only records are a documented difference:
    # global_lesson has no maturity, so only the hub maps confidence -> rank.)
    for maturity in ("L0", "L1", "L2", "L3"):
        for conf in (1, 3):
            h = HubRecord(id="X", schema="memory_item", path="x",
                          fields={"maturity": maturity, "evidence": {"confirmations": conf}},
                          body="").strength()
            c = CuratorItem(id="X", maturity=maturity, confirmations=conf).strength()
            assert h == c, f"strength drift at maturity={maturity} conf={conf}: hub={h} consumer={c}"
