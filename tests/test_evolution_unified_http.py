"""Exercise the unified HTTP registry with real MCP transport and isolated local state."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from test_evolution_interfaces import tool_value
from test_evolution_service import Scenario
from test_evolution_service import git_bin as shared_git_bin
from test_evolution_skill_workflow import configure_methods

git_bin = shared_git_bin
TOKEN = "fixture-http-token"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json, text/event-stream",
}


@pytest.fixture
def unified(tmp_path, git_bin, monkeypatch):
    scenario = Scenario(tmp_path, git_bin)
    configure_methods(scenario.service, tmp_path)
    workspace = scenario.service.workspace_root
    artifacts = tmp_path / "artifacts"
    workspace.mkdir(parents=True)
    artifacts.mkdir()
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "kernel": {
                    "repo_path": str(scenario.repo),
                    "repo_id": "fixture-kernel",
                    "owners": {"**/*.c": "owner"},
                    "page_size": 1,
                    "max_pages": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    for name, value in {
        "HMOPT_EVOLUTION_ROOT": scenario.service.store.root,
        "HMOPT_EVOLUTION_GIT_BIN": git_bin,
        "HMOPT_EVOLUTION_WORKSPACE_ROOT": workspace,
        "HMOPT_EVOLUTION_ARTIFACTS_ROOT": artifacts,
        "HMOPT_EVOLUTION_DISCOVERY_PROFILES": profiles,
        "HMOPT_MCP_ALLOWED_HOSTS": "testserver,localhost",
    }.items():
        monkeypatch.setenv(name, str(value))
    from hmopt.api import mcp_service
    from hmopt.api.mcp_server import create_app

    mcp_service.build_fastmcp_server.cache_clear()
    server = mcp_service.build_fastmcp_server()
    assert server is not None
    application = create_app(server, api_key=TOKEN, mount_path="/platform/mcp/")
    # Index retrieval is the only stub: no LLM/vector/Neo4j service is needed.
    monkeypatch.setattr(
        mcp_service,
        "_call_general_tool",
        lambda arguments, **kwargs: f"Retrieved kernel context: {arguments['query']}",
    )
    yield application, server, scenario, profiles
    mcp_service.build_fastmcp_server.cache_clear()


def rpc(client, method, params=None, *, headers=None):
    response = client.post(
        "/platform/mcp/",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
        headers=HEADERS if headers is None else headers,
    )
    assert response.status_code == 200, response.text
    value = response.json()
    assert "error" not in value, value
    return value["result"]


def rpc_value(result):
    assert not result.get("isError"), result
    if result.get("structuredContent") is not None:
        value = result["structuredContent"]
        return value["result"] if set(value) == {"result"} else value
    return json.loads(result["content"][0]["text"])


def legacy(client, tool, arguments=None):
    return client.post(
        "/tools/call",
        headers=HEADERS,
        json={"tool": tool, "arguments": arguments or {}},
    )


def test_http_lifespan_health_and_legacy_use_one_registry(unified, monkeypatch):
    application, server, scenario, _ = unified
    from hmopt.api import mcp_server

    def unexpected_build():
        raise AssertionError("A request must never rebuild the MCP registry")

    monkeypatch.setattr(mcp_server, "build_fastmcp_server", unexpected_build)
    assert application.state.mcp_server is server
    with TestClient(application) as client:
        initialized = rpc(
            client,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "http-integration-fixture", "version": "1"},
            },
        )
        assert initialized["serverInfo"]["name"] == "hmopt-kernel-index"
        names = {tool["name"] for tool in rpc(client, "tools/list")["tools"]}
        assert len(names) == 43
        assert {"kernel_index_code", "evolution_run_discovery", "evolution_submit"} <= names
        health = client.get("/health").json()
        assert set(health["tools"]) == names
        assert health["tool_name"] == "kernel_index_code"
        assert len(health["tool_names"]) == 5
        assert health["mcp_mount_path"] == "/platform/mcp"
        assert health["mcp_protocol_enabled"] is True
        assert health["mcp_api_key_required"] is True

        arguments = {"candidate_id": scenario.candidate_id}
        standard = rpc_value(
            rpc(client, "tools/call", {"name": "evolution_show", "arguments": arguments})
        )
        adapted = legacy(client, "evolution_show", arguments)
        assert adapted.status_code == 200, adapted.text
        assert adapted.json() == {"result": {"content": standard, "tool": "evolution_show"}}
        assert standard == scenario.service.store.read("candidate", scenario.candidate_id)
        listed = legacy(client, "evolution_list").json()["result"]["content"]
        assert listed == [standard]

        index = legacy(client, "kernel_index_code", {"query": "target"})
        assert index.json() == {
            "result": {
                "content": "Retrieved kernel context: target",
                "tool": "kernel_index_code",
            }
        }
        assert client.post("/platform/mcp/mcp", headers=HEADERS).status_code == 404


def test_bearer_protects_both_http_paths_and_operator_gates_still_apply(unified):
    application, _, scenario, _ = unified
    with TestClient(application) as client:
        for path in ("/tools/call", "/platform/mcp", "/platform/mcp/"):
            for headers in ({}, {"Authorization": "Bearer incorrect"}):
                response = client.post(path, json={}, headers=headers)
                assert response.status_code == 401
        assert client.get("/health").status_code == 200
        arguments = {
            "candidate_id": scenario.candidate_id,
            "action": "confirm",
            "actor": "owner",
            "expected_version": scenario.row["version"],
            "request_id": "http-cannot-approve",
            "payload": {"note": "The agent may not act as the owner."},
        }
        denied = legacy(client, "evolution_submit", arguments)
        assert denied.status_code == 400
        assert "operator CLI" in denied.text
        standard = rpc(client, "tools/call", {"name": "evolution_submit", "arguments": arguments})
        assert standard["isError"] is True
        assert scenario.service.store.read("candidate", scenario.candidate_id) == scenario.row
        extra = legacy(client, "evolution_list", {"root": "/not-operator-configured"})
        assert extra.status_code == 400
        assert "Unexpected tool arguments" in extra.text
        assert legacy(client, "evolution_confirm").status_code == 404


@pytest.mark.parametrize("error_type,status", [(RuntimeError, 500), (ValueError, 400)])
def test_legacy_preserves_internal_error_and_invalid_argument_statuses(
    unified, monkeypatch, error_type, status
):
    application, _, _, _ = unified
    from hmopt.api import mcp_service

    def failing_index(*args, **kwargs):
        raise error_type("Fixture retrieval failure")

    monkeypatch.setattr(mcp_service, "_call_general_tool", failing_index)
    with TestClient(application) as client:
        result = legacy(client, "kernel_index_code", {"query": "target"})
        assert result.status_code == status
        assert "Fixture retrieval failure" in result.text


@pytest.mark.parametrize("arguments", [[], "", False, 0])
def test_legacy_rejects_nonobject_arguments_even_when_empty(unified, arguments):
    application, _, _, _ = unified
    with TestClient(application) as client:
        response = client.post(
            "/tools/call",
            headers=HEADERS,
            json={"tool": "evolution_list", "arguments": arguments},
        )
        assert response.status_code == 400
        assert "arguments must be an object" in response.text


def test_http_configuration_is_frozen_and_reads_share_the_configured_store(unified, monkeypatch):
    application, _, scenario, profiles = unified
    with TestClient(application) as client:
        original = legacy(client, "evolution_discovery_profiles").json()["result"]["content"]
        assert original["store_root"] == str(scenario.service.store.root)
        assert [profile["name"] for profile in original["profiles"]] == ["kernel"]
        profiles.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("HMOPT_EVOLUTION_ROOT", str(profiles.parent / "wrong-state"))
        subsequent = rpc_value(
            rpc(client, "tools/call", {"name": "evolution_discovery_profiles", "arguments": {}})
        )
        assert subsequent == original
        with scenario.service.store.transaction() as db:
            row = scenario.service.store.put(
                db, "batch", "shared-http-fixture", {"status": "partial"}
            )
        read = legacy(client, "evolution_read", {"kind": "batch", "record_id": row["id"]})
        assert read.json()["result"]["content"] == row
        assert not (profiles.parent / "wrong-state").exists()


@pytest.mark.filterwarnings("ignore:Use `streamable_http_client` instead.:DeprecationWarning")
def test_official_mcp_http_sdk_initializes_and_calls_unified_tools(unified):
    application, _, scenario, _ = unified

    def client_factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=application), **kwargs)

    async def exercise():
        # ASGITransport keeps this real SDK/HTTP round-trip local, with no external service.
        # This SDK entrypoint is available across the declared mcp>=1.16,<2 range.
        async with (
            application.router.lifespan_context(application),
            streamablehttp_client(
                "http://localhost/platform/mcp/",
                headers=HEADERS,
                httpx_client_factory=client_factory,
            ) as (reader, writer, _),
            ClientSession(reader, writer) as session,
        ):
            initialized = await session.initialize()
            assert initialized.serverInfo.name == "hmopt-kernel-index"
            tools = (await session.list_tools()).tools
            assert len(tools) == 43
            row = tool_value(
                await session.call_tool("evolution_show", {"candidate_id": scenario.candidate_id})
            )
            assert row == scenario.row
            from test_evolution_skill_workflow import exercise_skill_mcp

            await exercise_skill_mcp(session, scenario)
            index = await session.call_tool("kernel_index_code", {"query": "target"})
            assert not index.isError
            assert index.content[0].text == "Retrieved kernel context: target"

    asyncio.run(exercise())
