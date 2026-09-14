"""Lightweight CLI commands must remain usable without the legacy runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Run in a fresh interpreter so modules loaded by other tests cannot hide an
# accidental eager import. Refuse both legacy runtime imports and connections.
_CLI_PROBE = """
import importlib.abc
import json
import runpy
import sys

blocked = (
    "hmopt.indexing", "hmopt.analysis", "hmopt.orchestration", "hmopt.storage",
    "hmopt.core.llm", "openai", "neo4j", "llama_index",
)

class RefuseHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked):
            raise AssertionError("lightweight command imported " + fullname)

def refuse_connections(event, args):
    if event == "socket.connect":
        raise AssertionError("lightweight command attempted a network connection")

sys.meta_path.insert(0, RefuseHeavyImports())
sys.addaudithook(refuse_connections)
sys.argv = ["hmopt", *json.loads(sys.argv[1])]
runpy.run_module("hmopt.cli", run_name="__main__")
"""


def _run_probe(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("HMOPT_LLM_") or name in {"OPENAI_API_KEY", "OPENAI_BASE_URL"}:
            env.pop(name)
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
        }
    )
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script, *arguments],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--help"], "evolve"),
        (["list-pipeline-profiles"], "generic_full"),
        (["list-pipeline-profiles", "--as-json"], '"hyperhold_full"'),
        (["evolve", "--help"], "Usage"),
        (["evolve", "schema", "plan"], '"candidate_id"'),
    ],
)
def test_lightweight_cli_without_model_or_legacy_runtime(arguments, expected):
    result = _run_probe(_CLI_PROBE, json.dumps(arguments))
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout


def test_prompting_import_does_not_load_other_agents_or_legacy_runtime():
    script = """
import sys
from hmopt.agents.prompting import render_prompt_template
import hmopt.agents as agents

assert render_prompt_template("missing.md", "Hello {value}", value="kernel") == "Hello kernel"
assert "ReviewerAgent" in dir(agents)
assert "hmopt.agents.trace_analyst" not in sys.modules
assert "hmopt.agents.reviewer" not in sys.modules
assert not any(name == "hmopt.analysis" or name.startswith("hmopt.analysis.") for name in sys.modules)
try:
    agents.not_an_agent
except AttributeError:
    pass
else:
    raise AssertionError("unknown exports must raise AttributeError")
"""
    result = _run_probe(script)
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_configuration_exports_are_lazy_and_preserve_import_identity():
    script = """
import sys
import hmopt.core as core
from hmopt.core import AppConfig, ConfigError
from hmopt.core.config import AppConfig as DirectConfig

assert AppConfig is DirectConfig is core.AppConfig
assert ConfigError is core.ConfigError
assert set(core.__all__) <= set(dir(core))
assert 'hmopt.core.llm' not in sys.modules
assert 'hmopt.core.run_context' not in sys.modules
assert 'hmopt.storage' not in sys.modules
try:
    core.not_an_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown exports must raise AttributeError')
from hmopt.core import LLMClient, RunContext
from hmopt.core.llm import LLMClient as DirectClient
from hmopt.core.run_context import RunContext as DirectContext
assert LLMClient is DirectClient
assert RunContext is DirectContext
"""
    result = _run_probe(script)
    assert result.returncode == 0, result.stdout + result.stderr
