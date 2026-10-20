"""Explicit model-response fixtures; these do not measure a model's reasoning quality."""

from hmopt.evolution.change_analysis import _changed_lines


def no_pattern_report(prepared):
    packet = prepared["packet"]
    findings = []
    for file in packet["files"]:
        citations = []
        for side in ("before", "after"):
            content = file[side].get("content", "")
            if not content.strip():
                continue
            lines = content.splitlines()
            changed = _changed_lines(file["patch"])[side]
            number = min(changed) if changed else 1
            line_numbers = file[side].get("line_numbers")
            index = line_numbers.index(number) if line_numbers else number - 1
            citations.append(
                {
                    "path": file["path"],
                    "side": side,
                    "line_start": number,
                    "quote": lines[index],
                }
            )
        if citations:
            findings.append(
                {
                    "paths": [file["path"]],
                    "what_changed": "Fixture analysis records this concrete file change.",
                    "how_behavior_changes": "Fixture response does not assert a reusable improvement.",
                    "why": "The fixture supplies no established reusable optimization mechanism.",
                    "why_status": "unknown",
                    "citations": citations,
                }
            )
    return {
        "source_id": packet["source_id"],
        "packet_sha256": prepared["job"]["data"]["packet_sha256"],
        "outcome": "no_pattern",
        "summary": "Explicit no-pattern model response fixture.",
        "findings": findings,
        "proposals": [],
        "unknowns": [],
    }


def complete_fixture_history(service, repo):
    from hmopt.evolution.change_analysis import (
        HistoryAnalysis,
        analysis_backlog,
        prepare_analysis,
        submit_analysis,
    )

    for job in analysis_backlog(service, repo)["jobs"]:
        prepared = prepare_analysis(service, repo, job["id"])
        submit_analysis(
            service,
            HistoryAnalysis.model_validate(no_pattern_report(prepared)),
            actor="fixture-researcher",
            expected_version=prepared["job"]["version"],
            request_id="fixture-analysis:" + job["id"],
        )
