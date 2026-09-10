"""CLI and real local stdio protocol tests; no network, model or device calls."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from test_evolution_service import Scenario
from typer.testing import CliRunner

from hmopt.evolution.cli import app as evolution_app
from hmopt.evolution.store import digest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()


@pytest.fixture(params=["standalone", "nested"])
def cli(request):
    if request.param == "standalone":
        return evolution_app, []
    from hmopt.cli import app

    return app, ["evolve"]


@pytest.fixture
def scenario(tmp_path):
    binary = shutil.which("git")
    bundled = (
        Path.home()
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
        / "native/git/cmd/git.exe"
    )
    if not binary and bundled.is_file():
        binary = str(bundled)
    if not binary:
        pytest.skip("A local Git executable is needed for the interface fixture")
    return Scenario(tmp_path, binary)


def invoke(cli, root, *arguments, git_bin="git"):
    app, prefix = cli
    return RUNNER.invoke(
        app,
        [*prefix, "--root", str(root), "--git-bin", git_bin, *arguments],
        env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "160"},
    )


def success(result):
    assert result.exit_code == 0, result.output or repr(result.exception)
    return json.loads(result.stdout)


def user_error(result, text):
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert text.casefold() in result.output.casefold(), result.output
    assert "Traceback" not in result.output


def test_help_and_schema_work_for_standalone_and_nested_entrypoints(cli, tmp_path):
    root = tmp_path / "unused-state"
    help_result = invoke(cli, root, "--help")
    assert help_result.exit_code == 0, help_result.output
    for name in ("mcp-stdio", "check-contract", "schema", "decide", "validate"):
        assert name in help_result.output
    for kind, field in (
        ("history", "source_id"),
        ("hotspot", "revision"),
        ("pattern", "matcher"),
        ("plan", "validation"),
        ("review", "subject_digest"),
        ("ab-report", "baseline"),
    ):
        schema = success(invoke(cli, root, "schema", kind))
        assert schema["type"] == "object"
        assert field in schema["properties"]
    user_error(invoke(cli, root, "schema", "unknown"), "Contract kind")
    assert not root.exists(), "Help and schema export must not initialize state"


@pytest.mark.parametrize(
    ("content", "arguments", "message"),
    [
        ("{", ["import-history"], "Expecting"),
        ("{}", ["import-history"], "JSON array"),
        ("[{}]", ["import-history"], "source_id"),
        (
            "[]",
            [
                "decide",
                "missing",
                "confirm",
                "--actor",
                "owner",
                "--version",
                "1",
                "--request-id",
                "bad-payload",
                "--note",
                "An explanatory decision note.",
                "--payload",
            ],
            "object",
        ),
    ],
)
def test_cli_invalid_json_and_contracts_are_actionable_errors(
    cli, tmp_path, content, arguments, message
):
    path = tmp_path / "input.json"
    path.write_text(content, encoding="utf-8")
    result = invoke(cli, tmp_path / "state", *arguments, str(path))
    user_error(result, message)


def test_cli_initialization_import_and_reads_share_persistent_state(cli, tmp_path):
    root = tmp_path / "state"
    initialized = success(invoke(cli, root, "init"))
    assert initialized["root"] == str(root.resolve())
    record = {
        "source_id": "cli-history",
        "repo_id": "reference-repo",
        "revision": "a" * 40,
        "subject": "Reference review",
        "message": "Imported notes are evidence only.",
        "paths": ["kernel.c"],
        "patch": "",
        "review_notes": ["Review exported for inspection."],
    }
    path = tmp_path / "history.json"
    path.write_text(json.dumps([record]), encoding="utf-8-sig")
    imported = success(invoke(cli, root, "import-history", str(path)))
    assert imported["changes"] == 1
    listed = success(invoke(cli, root, "list", "--kind", "history"))
    shown = success(invoke(cli, root, "show", "cli-history", "--kind", "history"))
    assert listed == [shown]
    assert shown["data"]["review_notes"] == record["review_notes"]


@pytest.mark.parametrize("hotspot_value", [None, 42, True, {}])
def test_cli_hotspot_input_requires_an_array(cli, scenario, tmp_path, hotspot_value):
    owners = tmp_path / "owners.json"
    owners.write_text(json.dumps({"**/*.c": "owner"}), encoding="utf-8")
    hotspots = tmp_path / "hotspots.json"
    hotspots.write_text(json.dumps(hotspot_value), encoding="utf-8")
    result = invoke(
        cli,
        scenario.service.store.root,
        "scan",
        str(scenario.repo),
        "--owners",
        str(owners),
        "--hotspots",
        str(hotspots),
        git_bin=scenario.git_bin,
    )
    user_error(result, "array")


def test_cli_handoff_export_preserves_existing_files(cli, scenario, tmp_path):
    s = scenario
    s.confirm()
    path = tmp_path / "handoff.json"
    packet = success(
        invoke(
            cli,
            s.service.store.root,
            "handoff",
            s.candidate_id,
            "--output",
            str(path),
            git_bin=s.git_bin,
        )
    )
    assert packet["role"] == "architect"
    assert json.loads(path.read_text(encoding="utf-8")) == packet
    previous = path.read_bytes()
    user_error(
        invoke(
            cli,
            s.service.store.root,
            "handoff",
            s.candidate_id,
            "--output",
            str(path),
            git_bin=s.git_bin,
        ),
        "exist",
    )
    assert path.read_bytes() == previous


def test_check_contract_digest_matches_plan_review_and_gate_errors_are_clean(
    cli, scenario, tmp_path
):
    s = scenario
    plan = s.plan_payload()["plan"]
    # Normalization must supply the same defaults later used by approve_plan.
    del plan["validation"]["minimum_pairs"]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    checked = success(invoke(cli, s.service.store.root, "check-contract", "plan", str(path)))
    assert checked["sha256"] == digest(checked["contract"])
    assert checked["contract"]["validation"]["minimum_pairs"] == 3
    version = s.row["version"]
    denied = invoke(
        cli,
        s.service.store.root,
        "decide",
        s.candidate_id,
        "confirm",
        "--actor",
        "not-owner",
        "--version",
        str(version),
        "--request-id",
        "denied-owner",
        "--note",
        "Attempt an owner-only decision.",
        git_bin=s.git_bin,
    )
    user_error(denied, "assigned owner")
    assert s.service.store.read("candidate", s.candidate_id) == s.row
    s.confirm()
    payload = {"plan": checked["contract"], "review": s.review(checked["sha256"], "plan-reviewer")}
    path.write_text(json.dumps(payload), encoding="utf-8")
    approved = success(
        invoke(
            cli,
            s.service.store.root,
            "decide",
            s.candidate_id,
            "approve_plan",
            "--actor",
            "plan-reviewer",
            "--version",
            str(s.row["version"]),
            "--request-id",
            "cli-plan",
            "--payload",
            str(path),
            git_bin=s.git_bin,
        )
    )
    assert approved["data"]["plan_digest"] == checked["sha256"]


_STDIO_MAIN = """
import runpy
import socket
import sys
def no_connections(event, args):
    if event == 'socket.connect':
        # Windows asyncio implements its internal wake-up socketpair using a
        # loopback connection. Permit only that standard-library call site.
        caller = sys._getframe(1)
        if (caller.f_code.co_name == '_fallback_socketpair'
                and caller.f_code.co_filename == socket.__file__):
            return
        raise AssertionError('Local stdio workflow attempted a network connection')
sys.addaudithook(no_connections)
runpy.run_module('hmopt.cli', run_name='__main__')
"""


@asynccontextmanager
async def local_session(scenario):
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith("HMOPT_LLM_") or name in {"OPENAI_API_KEY", "OPENAI_BASE_URL"}:
            env.pop(name)
    env.update(
        PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1", TERM="dumb"
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-X",
            "utf8",
            "-c",
            _STDIO_MAIN,
            "evolve",
            "--root",
            str(scenario.service.store.root),
            "--git-bin",
            scenario.git_bin,
            "mcp-stdio",
        ],
        cwd=str(REPO_ROOT),
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=15)
        ) as session:
            initialized = await session.initialize()
            assert initialized.serverInfo.name == "hmopt-evolution"
            yield session


def tool_value(result):
    assert not result.isError, result.content
    value = result.structuredContent
    if value is None:
        # Bare dict return annotations use text JSON in supported FastMCP
        # versions; typed list outputs also expose structuredContent.
        texts = [item.text for item in result.content if item.type == "text"]
        assert len(texts) == 1, result.content
        return json.loads(texts[0])
    return value["result"] if set(value) == {"result"} else value


def tool_error(result, message):
    assert result.isError, result
    text = " ".join(item.text for item in result.content if item.type == "text")
    assert message.casefold() in text.casefold(), text


def test_stdio_tool_registry_preserves_operator_boundaries_and_capture_is_unverified(scenario):
    s = scenario

    async def exercise():
        async with local_session(s) as session:
            schemas = {tool.name: tool.inputSchema for tool in (await session.list_tools()).tools}
            assert set(schemas) == {
                "evolution_list",
                "evolution_show",
                "evolution_handoff",
                "evolution_submit",
                "evolution_validate",
                "evolution_recall",
                "evolution_audit",
                "evolution_evidence",
                "evolution_capture",
                "evolution_quality",
                "evolution_convert_ic",
                "evolution_dispatch",
            }
            assert "allow_synthetic" not in schemas["evolution_validate"]["properties"]
            assert "simulation" not in schemas["evolution_validate"]["properties"]
            for action in ("confirm", "reject", "retry_validation", "promote", "activate_pattern"):
                result = await session.call_tool(
                    "evolution_submit",
                    {
                        "candidate_id": s.candidate_id,
                        "action": action,
                        "actor": "owner",
                        "expected_version": 1,
                        "request_id": "forbidden-" + action,
                        "payload": {},
                    },
                )
                tool_error(result, "operator CLI")
            tool_error(
                await session.call_tool("evolution_handoff", {"candidate_id": s.candidate_id}),
                "No executable handoff",
            )
            tool_error(
                await session.call_tool(
                    "evolution_dispatch",
                    {
                        "candidate_id": s.candidate_id,
                        "actor": "agent",
                        "request_id": "unconfirmed-dispatch",
                    },
                ),
                "No executable handoff",
            )
            assert "output" not in schemas["evolution_dispatch"]["properties"]
            quality = tool_value(await session.call_tool("evolution_quality", {}))
            assert quality["patterns"][s.pattern_key]["owner_undecided"] == 1
            tool_error(
                await session.call_tool("evolution_submit", {"candidate_id": s.candidate_id}),
                "required",
            )
            tool_error(await session.call_tool("evolution_promote", {}), "Unknown tool")
            original = tool_value(
                await session.call_tool("evolution_show", {"candidate_id": s.candidate_id})
            )
            assert original == s.row
            captured = tool_value(
                await session.call_tool(
                    "evolution_capture",
                    {
                        "signal": "expert_decision",
                        "source_id": s.candidate_id,
                        "actor": "agent",
                        "recipe": "Untrusted fixture claims approval; it cannot authorize implementation.",
                    },
                )
            )
            assert captured["data"]["tier"] == "journal"
            assert captured["data"]["outcome"] == "unverified"
            assert captured["data"]["eligible_for_promotion"] is False
            assert s.service.store.read("candidate", s.candidate_id) == original
            evidence = tool_value(
                await session.call_tool(
                    "evolution_evidence",
                    {
                        "sha256": captured["data"]["evidence"],
                    },
                )
            )
            assert digest(evidence["content"]) == evidence["sha256"]
            assert evidence["content"] == original
        # A second actual server process reads the same persisted journal.
        async with local_session(s) as session:
            skills = tool_value(await session.call_tool("evolution_list", {"kind": "skill"}))
            assert skills == [captured]

    asyncio.run(exercise())


def test_stdio_and_cli_share_gates_reviews_and_nonhardware_validation(scenario):
    s = scenario

    async def exercise():
        async with local_session(s) as session:

            async def submit(action, actor, payload, request_id):
                return await session.call_tool(
                    "evolution_submit",
                    {
                        "candidate_id": s.candidate_id,
                        "action": action,
                        "actor": actor,
                        "expected_version": s.row["version"],
                        "request_id": request_id,
                        "payload": payload,
                    },
                )

            tool_error(
                await submit("approve_plan", "plan-reviewer", s.plan_payload(), "too-early"),
                "owner confirmation",
            )
            # Confirm through the actual nested CLI while the MCP server remains running.
            from hmopt.cli import app

            s.row = success(
                invoke(
                    (app, ["evolve"]),
                    s.service.store.root,
                    "decide",
                    s.candidate_id,
                    "confirm",
                    "--actor",
                    "owner",
                    "--version",
                    "1",
                    "--request-id",
                    "operator-confirm",
                    "--note",
                    "Owner confirms this fixture.",
                    git_bin=s.git_bin,
                )
            )
            shown = tool_value(
                await session.call_tool("evolution_show", {"candidate_id": s.candidate_id})
            )
            assert shown == s.row
            handoff = tool_value(
                await session.call_tool("evolution_handoff", {"candidate_id": s.candidate_id})
            )
            assert handoff["role"] == "architect" and not handoff["source_changes_allowed"]

            plan_payload = s.plan_payload()
            approved = tool_value(
                await submit("approve_plan", "plan-reviewer", plan_payload, "plan")
            )
            assert (
                tool_value(await submit("approve_plan", "plan-reviewer", plan_payload, "plan"))
                == approved
            )
            s.row = approved
            handoff = tool_value(
                await session.call_tool("evolution_handoff", {"candidate_id": s.candidate_id})
            )
            assert handoff["role"] == "implementer" and handoff["source_changes_allowed"]
            (s.repo / s.path).write_text(
                "int target(int x) { return cached_lookup(x); }\n", encoding="utf-8"
            )
            revision = s.commit("interface fixture implementation")
            s.row = tool_value(
                await submit(
                    "record_implementation", "implementer", {"revision": revision}, "implementation"
                )
            )
            subject = s.row["data"]["implementation_digest"]
            tool_error(
                await submit(
                    "approve_code", "implementer", s.review(subject, "implementer"), "self-review"
                ),
                "independent reviewer",
            )
            s.row = tool_value(
                await submit(
                    "approve_code",
                    "code-reviewer",
                    s.review(subject, "code-reviewer"),
                    "code-review",
                )
            )

            result = await session.call_tool(
                "evolution_validate",
                {
                    "candidate_id": s.candidate_id,
                    "report": s.report_data(hardware=False),
                    "actor": "validator",
                    "expected_version": s.row["version"],
                    "request_id": "nonhardware",
                },
            )
            s.row = tool_value(result)
            validation = s.row["data"]["validation"]
            assert s.row["data"]["stage"] == "code_approved"
            assert validation["result"]["verdict"] == "inconclusive"
            assert validation["simulation"] is True
            assert validation["eligible_for_promotion"] is False
            assert s.service.store.read("candidate", s.candidate_id) == s.row
            journal = tool_value(await session.call_tool("evolution_recall", {"query": s.path}))
            assert journal and all(
                row["data"]["eligible_for_promotion"] is False for row in journal
            )
            audit = tool_value(
                await session.call_tool("evolution_audit", {"candidate_id": s.candidate_id})
            )
            assert [event["action"] for event in audit].count("approve_plan") == 1
            assert audit[-1]["action"] == "validate"

    asyncio.run(exercise())
