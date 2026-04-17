from hmopt.agents.reviewer import ReviewerAgent
from hmopt.agents.safety import SafetyGuard


class _FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply

    def chat(self, messages):
        return self.reply


def test_reviewer_agent_parses_explicit_approve():
    agent = ReviewerAgent(_FakeLLM("Decision: APPROVE\nRisks: low\nRationale: ok"), SafetyGuard())
    decision = agent.review(
        evidence_summary="summary",
        patch_summary="patch",
        build_success=True,
        test_success=True,
        iteration=1,
    )
    assert decision["decision"] == "approve"
    assert decision["risk_summary"] == "low"


def test_reviewer_agent_defaults_to_reject_on_failed_verification_without_explicit_decision():
    agent = ReviewerAgent(_FakeLLM("Rationale: verification is incomplete"), SafetyGuard())
    decision = agent.review(
        evidence_summary="summary",
        patch_summary="patch",
        build_success=False,
        test_success=False,
        iteration=1,
    )
    assert decision["decision"] == "reject"
