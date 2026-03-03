"""FastMCP service wrapper for Git MCP tools."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import git

from hmopt.mcp_server_git.server import (
    DEFAULT_CONTEXT_LINES,
    git_add,
    git_branch,
    git_checkout,
    git_commit,
    git_create_branch,
    git_diff,
    git_diff_staged,
    git_diff_unstaged,
    git_init,
    git_log,
    git_reset,
    git_show,
    git_status,
)

logger = logging.getLogger(__name__)
DEFAULT_SERVER_NAME = "hmopt-git-mcp"


def get_default_repository() -> str | None:
    value = os.getenv("HMOPT_GIT_MCP_REPOSITORY", "").strip()
    return value or None


def _resolve_repo_path(repo_path: str | None) -> str:
    resolved = (repo_path or "").strip()
    if resolved:
        return resolved

    default_repo = get_default_repository()
    if default_repo:
        return default_repo

    raise ValueError(
        "repo_path is required. Provide tool argument repo_path or set HMOPT_GIT_MCP_REPOSITORY."
    )


def _resolve_repo(repo_path: str | None) -> git.Repo:
    return git.Repo(Path(_resolve_repo_path(repo_path)))


@lru_cache(maxsize=1)
def build_git_fastmcp_server() -> Any | None:
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError:
        logger.warning(
            "mcp package is not installed; git MCP protocol endpoint is unavailable. "
            "Install with: pip install 'mcp[cli]'"
        )
        return None

    server_name = os.getenv("HMOPT_GIT_MCP_SERVER_NAME", DEFAULT_SERVER_NAME).strip() or DEFAULT_SERVER_NAME
    mcp = FastMCP(server_name, stateless_http=True, json_response=True)

    @mcp.tool(name="git_status", description="Shows working tree status for a git repository.")
    def mcp_git_status(repo_path: str | None = None) -> str:
        return git_status(_resolve_repo(repo_path))

    @mcp.tool(name="git_diff_unstaged", description="Shows unstaged changes in the working tree.")
    def mcp_git_diff_unstaged(
        context_lines: int = DEFAULT_CONTEXT_LINES,
        repo_path: str | None = None,
    ) -> str:
        return git_diff_unstaged(_resolve_repo(repo_path), context_lines)

    @mcp.tool(name="git_diff_staged", description="Shows staged changes.")
    def mcp_git_diff_staged(
        context_lines: int = DEFAULT_CONTEXT_LINES,
        repo_path: str | None = None,
    ) -> str:
        return git_diff_staged(_resolve_repo(repo_path), context_lines)

    @mcp.tool(name="git_diff", description="Shows differences against a branch/commit/tag target.")
    def mcp_git_diff(
        target: str,
        context_lines: int = DEFAULT_CONTEXT_LINES,
        repo_path: str | None = None,
    ) -> str:
        return git_diff(_resolve_repo(repo_path), target, context_lines)

    @mcp.tool(name="git_commit", description="Creates a git commit from staged changes.")
    def mcp_git_commit(message: str, repo_path: str | None = None) -> str:
        return git_commit(_resolve_repo(repo_path), message)

    @mcp.tool(name="git_add", description="Stages files to git index.")
    def mcp_git_add(files: list[str], repo_path: str | None = None) -> str:
        return git_add(_resolve_repo(repo_path), files)

    @mcp.tool(name="git_reset", description="Unstages all staged changes.")
    def mcp_git_reset(repo_path: str | None = None) -> str:
        return git_reset(_resolve_repo(repo_path))

    @mcp.tool(name="git_log", description="Shows git commit log.")
    def mcp_git_log(max_count: int = 10, repo_path: str | None = None) -> list[str]:
        return git_log(_resolve_repo(repo_path), max_count)

    @mcp.tool(name="git_create_branch", description="Creates a branch from current/base branch.")
    def mcp_git_create_branch(
        branch_name: str,
        base_branch: str | None = None,
        repo_path: str | None = None,
    ) -> str:
        return git_create_branch(_resolve_repo(repo_path), branch_name, base_branch)

    @mcp.tool(name="git_checkout", description="Switches to a given branch.")
    def mcp_git_checkout(branch_name: str, repo_path: str | None = None) -> str:
        return git_checkout(_resolve_repo(repo_path), branch_name)

    @mcp.tool(name="git_show", description="Shows details and patch of a revision.")
    def mcp_git_show(revision: str, repo_path: str | None = None) -> str:
        return git_show(_resolve_repo(repo_path), revision)

    @mcp.tool(name="git_init", description="Initializes a git repository.")
    def mcp_git_init(repo_path: str | None = None) -> str:
        return git_init(_resolve_repo_path(repo_path))

    @mcp.tool(name="git_branch", description="Lists branches by local/remote/all filters.")
    def mcp_git_branch(
        branch_type: str,
        contains: str | None = None,
        not_contains: str | None = None,
        repo_path: str | None = None,
    ) -> str:
        return git_branch(_resolve_repo(repo_path), branch_type, contains, not_contains)

    return mcp
