"""Real Git -> stdio MCP -> signed approval HTTP -> real correctness adapter -> journal.

Research and independent-review answers are explicit fixtures, not LLM quality evidence.
The adapter actually executes both frozen Python revisions; no device/mail/LLM is contacted.
"""

import asyncio
import json
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
from test_evolution_correctness import policy
from test_evolution_interfaces import local_session, tool_value
from test_evolution_mining import GIT, commit, git
from test_evolution_pattern_synthesis import change as shared_change
from test_evolution_pattern_synthesis import publish
from test_evolution_pattern_synthesis import repo as shared_repo
from test_evolution_pattern_synthesis import synthesis as shared_synthesis
from test_evolution_production import approval_config as shared_approval_config
from test_evolution_production import decision, post

from hmopt.api.evolution_mcp_service import build_evolution_fastmcp_server
from hmopt.api.mcp_server import create_app
from hmopt.evolution.approval import deliver_notifications
from hmopt.evolution.experiments import run_experiment
from hmopt.evolution.methods import code_context
from hmopt.evolution.production import ValidationRunnerConfig
from hmopt.evolution.research import prepare_research

repo = shared_repo
change = shared_change
synthesis = shared_synthesis
approval_config = shared_approval_config

ADAPTER = """import hashlib, json, os, subprocess, sys
from pathlib import Path
request = json.loads(Path(os.environ["HMOPT_EXPERIMENT_REQUEST"]).read_text(encoding="utf-8"))
policy = request["policy"]
report = {"kind": "correctness", "candidate_id": request["candidate"]["id"],
          "implementation_revision": request["implementation_revision"],
          "policy": policy, "execution_kind": "local"}
raw = {}
def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
for arm, revision in (("baseline", request["baseline_revision"]),
                      ("candidate", request["implementation_revision"])):
    source = subprocess.run([sys.argv[1], "-C", request["execution_repo"],
                             "show", revision + ":consumer.py"],
                            check=True, capture_output=True, timeout=20).stdout
    namespace = {}
    exec(compile(source, "consumer.py", "exec"), namespace)
    function = namespace["consume"]
    outcomes = {}
    for name, cases in {"reproducer": [("   ", 0)],
                        "regressions": [("", 0), ("+7", 7), (" 42 ", 42), ("x", "ValueError")]}.items():
        observations = []
        for argument, expected in cases:
            try:
                actual = function(argument)
            except ValueError:
                actual = "ValueError"
            observations.append({"input": argument, "expected": expected, "actual": actual})
        outcomes[name] = {"outcome": "pass" if all(x["expected"] == x["actual"] for x in observations) else "fail",
                          "evidence_sha256": sha(observations)}
        raw[arm + ":" + name] = observations
    report[arm] = {**{key: policy[key] for key in ("device_id", "workload_id", "workload_config_sha256", "environment_sha256")},
                   "repo_revision": revision, "artifact_sha256": hashlib.sha256(source).hexdigest(),
                   "checks": outcomes}
report["functional_passed"] = all(c["outcome"] == "pass" for c in report["candidate"]["checks"].values())
output = Path(os.environ["HMOPT_EXPERIMENT_RESULT"])
output.with_name("observations.json").write_text(json.dumps(raw), encoding="utf-8")
output.write_text(json.dumps({"report": report}), encoding="utf-8")
"""


def test_full_workflow_with_real_before_after_execution(
    synthesis, approval_config, tmp_path, monkeypatch
):
    service, repo, prepared, payload = synthesis
    key = publish(service, prepared, payload)["data"]["output_ids"][0]
    review = prepare_research(service, repo, "review", [key], actor="pattern-reviewer")
    publish(
        service,
        review,
        {
            "decision": "approve",
            "rationale": "Fixture independent review: two real diffs share a guard defect; strict parsers excluded.",
        },
        actor="pattern-reviewer",
        request_id="pattern-review",
    )
    service.activate_pattern(
        key, actor="curator", note="Reviewed two exemplars and explicit scope."
    )
    # Workbench artifacts are task state, outside the reviewed business change.
    (repo / ".gitignore").write_text("workbench/.opencode/local/\n", encoding="utf-8", newline="\n")
    source = (
        "def consume(raw: str):\n"
        '    """Blank string input means zero; invalid numeric text raises ValueError."""\n'
        "    return int(raw) if raw else 0\n"
    )
    (repo / "consumer.py").write_text(source, encoding="utf-8", newline="\n")
    baseline = commit(repo, "new consumer")
    runtime = tmp_path.parent / (tmp_path.name + "-walkthrough")
    runtime.mkdir()
    adapter_path = runtime / "correctness_adapter.py"
    adapter_path.write_text(ADAPTER, encoding="utf-8", newline="\n")
    adapter = ValidationRunnerConfig(
        command=[sys.executable, str(adapter_path), GIT],
        cwd=str(tmp_path),
        resource_id="local-python",
        timeout_seconds=30,
    )
    options = {
        "root": str(service.store.root),
        "git_bin": GIT,
        "workspace_root": str(service.workspace_root),
        "discovery_profiles": {"consumer": {"repo_path": str(repo), "owners": {"**": "owner"}}},
        "production": {
            "approval": approval_config.model_dump(mode="json"),
            "validation": adapter.model_dump(mode="json"),
        },
    }
    config_path = runtime / "fixture-config.json"
    config_path.write_text(
        json.dumps({"schema_version": 1, **options}), encoding="utf-8", newline="\n"
    )
    server = build_evolution_fastmcp_server(**options)
    monkeypatch.setenv(approval_config.signing_key_env, "fixture-key-" * 4)

    async def exercise():
        async with local_session(
            SimpleNamespace(service=service, git_bin=GIT),
            extra_env={"HMOPT_EVOLUTION_CONFIG": str(config_path)},
        ) as session:

            async def call(name, **arguments):
                return tool_value(await session.call_tool(name, arguments))

            scan = await call(
                "evolution_scan",
                action="start",
                profile="consumer",
                actor="researcher",
                request_id="walkthrough-scan",
            )
            while scan["data"]["status"] == "running":
                scan = await call(
                    "evolution_scan",
                    action="next",
                    scan_id=scan["id"],
                    expected_version=scan["version"],
                )
            assert scan["data"]["coverage_complete"] and scan["data"]["candidate_count"] == 1
            row = service.store.list("candidate")[0]
            candidate_id = row["id"]
            assessment = await call(
                "evolution_prepare_research",
                profile="consumer",
                method="assess",
                subject_ids=[candidate_id],
                actor="researcher",
            )
            context = code_context(service, repo, baseline, "consumer.py")
            citation = {
                "context_sha256": context["sha256"],
                "line_start": 1,
                "quote": "\n".join(context["lines"]),
            }
            await call(
                "evolution_submit_research",
                research_id=assessment["research"]["id"],
                actor="researcher",
                expected_version=assessment["research"]["version"],
                request_id="assessment",
                payload={
                    "result": "applicable",
                    "summary": "Fixture: the explicit string/default contract supports this remedy.",
                    "checks": [
                        {
                            "criterion_id": criterion["id"],
                            "result": "met",
                            "explanation": "Fixture: source contract defines blank as zero and invalid numbers as errors.",
                            "citations": [citation],
                        }
                        for criterion in assessment["research"]["data"]["inputs"]["criteria"]
                    ],
                },
            )
            request = await call(
                "evolution_request_approval",
                candidate_id=candidate_id,
                actor="coordinator",
                request_id="expert-request",
            )
            delivered = []
            deliver_notifications(
                service,
                approval_config,
                actor="operator",
                transport=lambda *args: delivered.append(args),
            )
            assert len(delivered) == 1
            with TestClient(create_app(server)) as client:
                response = post(client, "/evolution/approval/decide", decision(request))
                assert response.status_code == 200, response.text
                assert (
                    post(client, "/evolution/approval/decide", decision(request)).json()
                    == response.json()
                )
            row = await call("evolution_show", candidate_id=candidate_id)
            assert row["data"]["stage"] == "confirmed" and row["data"]["approval_id"]

            async def transition(action, actor, payload):
                nonlocal row
                row = await call(
                    "evolution_submit",
                    candidate_id=candidate_id,
                    action=action,
                    actor=actor,
                    expected_version=row["version"],
                    request_id=action,
                    payload=payload,
                )

            plan = {
                "candidate_id": candidate_id,
                "base_revision": baseline,
                "author": "architect",
                "hypothesis": "Normalize blank strings to zero while preserving numeric/error behavior.",
                "bottleneck": "reliability",
                "metric_rationale": "Reproducer and regressions measure correctness.",
                "allowed_paths": ["consumer.py"],
                "validation": policy(),
            }
            normalized = await call("evolution_digest", kind="plan", payload=plan)

            def review_for(subject, actor):
                return {
                    "candidate_id": candidate_id,
                    "subject_digest": subject,
                    "author": actor,
                    "decision": "approve",
                    "rationale": "Fixture independent review of exact contract and change scope.",
                }

            await transition(
                "approve_plan",
                "plan-reviewer",
                {
                    "plan": normalized["contract"],
                    "review": review_for(normalized["sha256"], "plan-reviewer"),
                },
            )
            (repo / "consumer.py").write_text(
                source.replace("if raw else", "if raw.strip() else"), encoding="utf-8", newline="\n"
            )
            revision = commit(repo, "normalize blank consumer input")
            await transition("record_implementation", "implementer", {"revision": revision})
            await transition(
                "approve_code",
                "code-reviewer",
                review_for(row["data"]["implementation_digest"], "code-reviewer"),
            )
            job = await call(
                "evolution_experiment",
                action="prepare",
                candidate_id=candidate_id,
                actor="validator",
                request_id="experiment",
            )
            job = job["experiment"]
            result = run_experiment(service, job["id"], adapter)
            assert result["data"]["status"] == "complete"
            dossier = await call("evolution_dossier", candidate_id=candidate_id)
            assert dossier["candidate"]["data"]["stage"] == "validated"
            assert dossier["linked_records"]["experiment"]["ids"] == [job["id"]]
            assert len(dossier["source_ids"]) == 2 and len(dossier["approvals"]) == 1
            validation = dossier["candidate"]["data"]["validation"]
            assert validation["result"]["verdict"] == "pass"
            assert validation["result"]["hardware_verified"] is False
            skill = service.store.list("skill")[0]
            assert skill["data"]["tier"] == "journal"
            service.promote(
                skill["id"],
                tier="staging",
                actor="curator",
                expected_version=skill["version"],
                request_id="curate",
                note="Reviewed actual local reproducer and regression results.",
            )
            assert run_experiment(service, job["id"], adapter) == result
            assert git(repo, "status", "--porcelain") == ""
            assert not service.store.list("experiment_claim")

    asyncio.run(exercise())
