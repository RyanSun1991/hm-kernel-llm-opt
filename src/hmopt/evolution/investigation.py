"""Model-directed read-only investigation over the job's exact before/after Git objects."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .change_analysis import HistoryAnalysis
from .methods import code_context
from .mining import _Git, _path


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation: Literal["read", "search"]
    side: Literal["before", "after"]
    path: str | None = None
    query: str | None = Field(default=None, min_length=2, max_length=200)
    start_line: int = Field(default=1, ge=1, le=10_000_000)
    line_count: int = Field(default=100, ge=1, le=200)
    reason: str = Field(min_length=10, max_length=1000)

    @field_validator("path")
    @classmethod
    def normalized_path(cls, value):
        return _path(value) if value is not None else None

    @model_validator(mode="after")
    def shape(self):
        if self.operation == "read" and (self.path is None or self.query is not None):
            raise ValueError("read requires a path and no search query")
        if self.operation == "search" and (
            self.query is None or any(c in self.query for c in "\x00\n\r")
        ):
            raise ValueError("search requires a single-line literal query")
        return self


class InvestigationReply(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    analysis: HistoryAnalysis | None = None
    context_requests: list[ContextRequest] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def exclusive(self):
        if (self.analysis is not None) == bool(self.context_requests):
            raise ValueError("Return either a final analysis or one to four evidence requests")
        return self


def resolve_requests(service, repo, packet, requests):
    if service.repo_identity(repo) != packet["repo_id"]:
        raise ValueError("Investigation repository identity does not match the frozen packet")
    answers = []
    for request in requests:
        revision = packet["parent_revision"] if request.side == "before" else packet["revision"]
        if revision is None:
            answers.append(
                {
                    "request": request.model_dump(),
                    "error": "No before revision exists for a root commit",
                }
            )
            continue
        try:
            if request.operation == "read":
                result = code_context(
                    service,
                    repo,
                    revision,
                    request.path,
                    start_line=request.start_line,
                    line_count=request.line_count,
                )
            else:
                git = _Git(Path(repo), service.git_bin)
                # Exit code 1 means no matches, not a failed read. Request-local argv only.
                raw, clipped = git.read(
                    [
                        "grep",
                        "--no-textconv",
                        "-I",
                        "-n",
                        "-F",
                        "-e",
                        request.query,
                        revision,
                        "--",
                        *([":(literal)" + request.path] if request.path else []),
                    ],
                    32768,
                    allowed_exit_codes=(0, 1),
                )
                lines = raw.decode("utf-8", errors="strict").splitlines()
                result = {
                    "kind": "literal_git_search",
                    "repo_id": packet["repo_id"],
                    "revision": revision,
                    "query": request.query,
                    "matches": lines[:100],
                    "truncated": clipped or len(lines) > 100,
                    "next_action": "Read exact file windows to obtain citable evidence.",
                }
                with service.store.transaction() as db:
                    result = {"sha256": service.store.evidence(db, result), **result}
            answers.append({"request": request.model_dump(), "result": result})
        except (ValueError, OSError, TimeoutError) as error:
            # Missing symbols/files are observable evidence gaps, never invented context.
            answers.append({"request": request.model_dump(), "error": str(error)[:1000]})
    return answers
