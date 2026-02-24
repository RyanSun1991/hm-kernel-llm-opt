import os
import uuid
from ..models.thought import Thought, ThoughtRequest, ThoughtResponse
from ..models.session import ThinkingSession, _parse_assumption_id


class UltraThinkService:
    """
    Service: Orchestrates the sequential thinking process
    Handles session lifecycle and coordinates between models and interface
    """

    def __init__(self) -> None:
        self._disable_logging = (
            os.environ.get("DISABLE_THOUGHT_LOGGING", "").lower() == "true"
        )
        self._sessions: dict[str, ThinkingSession] = {}
        # Per-session idempotency replay cache: session_id -> idempotency_key -> response
        self._idempotency_cache: dict[str, dict[str, ThoughtResponse]] = {}

    def _resolve_cross_session_assumption(
        self, scoped_id: str, current_session_id: str
    ) -> tuple[str | None, bool]:
        """
        Resolve a cross-session assumption reference
        """
        target_session_id, local_id = _parse_assumption_id(scoped_id)

        if target_session_id is None:
            return scoped_id, True

        if target_session_id not in self._sessions:
            return None, False

        target_session = self._sessions[target_session_id]
        if local_id not in target_session.all_assumptions:
            return None, False

        return local_id, True

    def _recommended_next_action(
        self, *, thought: Thought, unresolved: list[str], warnings: list[str]
    ) -> str:
        if not thought.next_thought_needed:
            return "finalize"
        if unresolved or warnings:
            return "resolve_assumptions"
        if thought.open_question:
            return "answer_open_question"
        if thought.steps and thought.current_step is not None and thought.current_step < len(thought.steps):
            return "advance_step"
        return "continue"

    def _build_response(
        self,
        *,
        session_id: str,
        thought: Thought,
        session: ThinkingSession,
        unresolved: list[str],
        warnings: list[str],
        duplicate_of_thought_number: int | None = None,
    ) -> ThoughtResponse:
        progress_ratio = min(1.0, thought.thought_number / max(1, thought.total_thoughts))
        return ThoughtResponse(
            session_id=session_id,
            thought_number=thought.thought_number,
            total_thoughts=thought.total_thoughts,
            next_thought_needed=thought.next_thought_needed,
            branches=session.branch_ids,
            thought_history_length=session.thought_count,
            confidence=thought.confidence,
            uncertainty_notes=thought.uncertainty_notes,
            outcome=thought.outcome,
            plan=thought.plan,
            next=thought.next,
            reflect=thought.reflect,
            steps=thought.steps,
            current_step=thought.current_step,
            rationale=thought.rationale,
            open_question=thought.open_question,
            request_id=thought.request_id,
            idempotency_key=thought.idempotency_key,
            duplicate_of_thought_number=duplicate_of_thought_number,
            progress_ratio=progress_ratio,
            recommended_next_action=self._recommended_next_action(
                thought=thought, unresolved=unresolved, warnings=warnings
            ),
            all_assumptions=session.all_assumptions,
            risky_assumptions=session.risky_assumptions,
            falsified_assumptions=session.falsified_assumptions,
            unresolved_references=unresolved,
            cross_session_warnings=warnings,
        )

    def process_thought(self, request: ThoughtRequest) -> ThoughtResponse:
        """Process a thought request."""
        if request.session_id is None:
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = ThinkingSession(disable_logging=self._disable_logging)
        else:
            session_id = request.session_id
            if session_id not in self._sessions:
                self._sessions[session_id] = ThinkingSession(disable_logging=self._disable_logging)

        session = self._sessions[session_id]

        if request.idempotency_key:
            cache = self._idempotency_cache.get(session_id, {})
            cached = cache.get(request.idempotency_key)
            if cached is not None:
                return cached.model_copy(update={"duplicate_of_thought_number": cached.thought_number})

        thought_number = request.thought_number if request.thought_number is not None else (session.thought_count + 1)
        next_thought_needed = (
            request.next_thought_needed
            if request.next_thought_needed is not None
            else (thought_number < request.total_thoughts)
        )

        validated_cross_session_refs: list[str] = []
        if request.depends_on_assumptions:
            for assumption_id in request.depends_on_assumptions:
                if ":" in assumption_id:
                    _, was_resolved = self._resolve_cross_session_assumption(assumption_id, session_id)
                    if was_resolved:
                        validated_cross_session_refs.append(assumption_id)

        thought_data = request.model_dump(exclude={"session_id"})
        thought_data["thought_number"] = thought_number
        thought_data["next_thought_needed"] = next_thought_needed
        thought = Thought(**thought_data)

        session.add_thought(thought, validated_cross_session_refs)

        unresolved = session.unresolved_references
        warnings = session.cross_session_warnings

        response = self._build_response(
            session_id=session_id,
            thought=thought,
            session=session,
            unresolved=unresolved,
            warnings=warnings,
        )

        if request.idempotency_key:
            self._idempotency_cache.setdefault(session_id, {})[request.idempotency_key] = response

        return response

    def get_session_snapshot(self, session_id: str, include_history: bool = True) -> dict:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"session not found: {session_id}")

        thoughts = []
        if include_history:
            thoughts = [
                {
                    "thought_number": t.thought_number,
                    "thought": t.thought,
                    "plan": t.plan,
                    "next": t.next,
                    "reflect": t.reflect,
                    "current_step": t.current_step,
                    "open_question": t.open_question,
                    "confidence": t.confidence,
                    "next_thought_needed": t.next_thought_needed,
                }
                for t in session.thoughts
            ]

        return {
            "session_id": session_id,
            "thought_history_length": session.thought_count,
            "branches": session.branch_ids,
            "assumption_count": len(session.all_assumptions),
            "risky_assumptions": session.risky_assumptions,
            "falsified_assumptions": session.falsified_assumptions,
            "unresolved_references": session.unresolved_references,
            "cross_session_warnings": session.cross_session_warnings,
            "thoughts": thoughts,
        }

    def list_sessions(self) -> list[dict]:
        items = []
        for sid, session in self._sessions.items():
            items.append(
                {
                    "session_id": sid,
                    "thought_history_length": session.thought_count,
                    "branch_count": len(session.branch_ids),
                    "assumption_count": len(session.all_assumptions),
                }
            )
        return sorted(items, key=lambda x: x["thought_history_length"], reverse=True)

    def reset_session(self, session_id: str, hard: bool = False) -> dict:
        if session_id not in self._sessions:
            return {"session_id": session_id, "existed": False, "reset": False}

        if hard:
            del self._sessions[session_id]
            self._idempotency_cache.pop(session_id, None)
            return {"session_id": session_id, "existed": True, "reset": True, "hard": True}

        self._sessions[session_id] = ThinkingSession(disable_logging=self._disable_logging)
        self._idempotency_cache.pop(session_id, None)
        return {"session_id": session_id, "existed": True, "reset": True, "hard": False}
