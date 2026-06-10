"""Conductor decision parsing must not be fooled by negations.

The old `"continue" in reply.lower()` check read "do not continue" /
"discontinue" as a continue signal and burned iterations on a fabricated
decision with a hard-coded next action.
"""
from hmopt.agents.conductor import _parse_decision, _parse_next_action


def test_explicit_decision_line_wins():
    assert _parse_decision("Decision: CONTINUE\nNext: hoist the loop") == "continue"
    assert _parse_decision("Decision: STOP\nNo further gains") == "stop"


def test_negated_continue_is_not_continue():
    assert _parse_decision("We should do not continue here.") == "stop"
    assert _parse_decision("Recommend to discontinue the loop.") == "stop"
    assert _parse_decision("Let's stop; diminishing returns.") == "stop"


def test_bare_continue_signal():
    assert _parse_decision("Keep going, another iteration looks worthwhile.") == "continue"


def test_ambiguous_or_offline_reply_fails_safe_to_stop():
    assert _parse_decision("[offline-llm:abc123] ...prompt echo...") == "stop"
    assert _parse_decision("") == "stop"


def test_next_action_parsed_from_reply_not_hardcoded():
    reply = "Decision: CONTINUE\nNext: eliminate the redundant sc->priority load"
    assert _parse_next_action(reply, "continue") == "eliminate the redundant sc->priority load"


def test_next_action_defaults_when_absent():
    assert _parse_next_action("Decision: STOP", "stop") == "report"
    assert _parse_next_action("Decision: CONTINUE", "continue") == "refine the identified hotspot"
