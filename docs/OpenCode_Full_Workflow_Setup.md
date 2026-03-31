# OpenCode Full Workflow Setup Guide

This document covers the end-to-end process for setting up the OpenCode multi-agent kernel optimization workflow — from installing the OpenCode binary to running the first analysis session.

---

## Prerequisites

- Linux machine (x86_64 or aarch64) with Docker installed
- Network access to download packages and Docker images
- A kernel source tree to analyze
- An LLM API endpoint (e.g., GLM-4.7 or compatible OpenAI-API-format model)

---

## Step 1: Install OpenCode

### 1.1 Download the OpenCode binary

Download the latest OpenCode release from GitHub:

```bash
# Check the latest release at: https://github.com/nicepkg/opencode/releases
# Download the appropriate binary for your platform

# Example for Linux x86_64:
curl -L -o opencode https://github.com/nicepkg/opencode/releases/latest/download/opencode-linux-amd64

# Example for Linux aarch64:
curl -L -o opencode https://github.com/nicepkg/opencode/releases/latest/download/opencode-linux-arm64
```

### 1.2 Install the binary

```bash
chmod +x opencode
sudo mv opencode /usr/local/bin/

# Verify installation
opencode --version
```

---

## Step 2: Configure OpenCode Global Settings

### 2.1 Create the OpenCode configuration file

```bash
mkdir -p ~/.config/opencode
vim ~/.config/opencode/opencode.json
```

### 2.2 Minimal configuration template

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-provider": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "local-provider",
      "options": {
        "baseURL": "<LLM_API_BASE_URL>/v1/",
        "apiKey": "<YOUR_API_KEY>"
      },
      "models": {
        "glm-47": {
          "id": "glm-4.7"
        }
      }
    }
  },
  "mcp": {
    "hmopt_kernel_index_remote": {
      "type": "remote",
      "enabled": true,
      "url": "http://127.0.0.1:7332/mcp/",
      "headers": {
        "Authorization": "<YOUR_MCP_API_KEY>"
      },
      "timeout": 30000
    }
  },
  "keybinds": {
    "app_exit": "ctrl+d,<leader>q"
  },
  "default_agent": "plan"
}
```

### 2.3 Configuration field reference

| Field | Description |
|-------|-------------|
| `provider.*.options.baseURL` | Your LLM API endpoint (must end with `/v1/`) |
| `provider.*.options.apiKey` | Your LLM API key |
| `provider.*.models.*.id` | Model ID supported by your API (e.g., `glm-4.7`) |
| `mcp.hmopt_kernel_index_remote.url` | HMOPT MCP server endpoint (default: `http://127.0.0.1:7332/mcp/`) |
| `mcp.hmopt_kernel_index_remote.headers.Authorization` | MCP server API key (if configured) |

### 2.4 Optional: Add additional MCP servers

You can register more MCP servers for extended capabilities:

```json
{
  "mcp": {
    "hmopt_kernel_index_remote": {
      "type": "remote",
      "enabled": true,
      "url": "http://127.0.0.1:7332/mcp/",
      "headers": { "Authorization": "<key>" },
      "timeout": 30000
    },
    "hmopt_git_mcp": {
      "type": "remote",
      "enabled": true,
      "url": "http://127.0.0.1:7334/mcp/",
      "timeout": 15000
    },
    "hmopt_build_mcp": {
      "type": "remote",
      "enabled": true,
      "url": "http://127.0.0.1:7335/mcp/",
      "timeout": 60000
    },
    "hmopt_seq_thinking": {
      "type": "remote",
      "enabled": true,
      "url": "http://127.0.0.1:7333/mcp/",
      "timeout": 30000
    }
  }
}
```

**Port summary for MCP services:**

| Service | Host Port | Container Port | Description |
|---------|-----------|----------------|-------------|
| Kernel Index MCP | 7332 | 7331 | Code retrieval (vector + graph) |
| Sequential Thinking MCP | 7333 | 7333 | Step-by-step reasoning |
| Git MCP | 7334 | 7334 | Git operations |
| Build MCP | 7335 | 7335 | Kernel build triggers |
| Auto-Test MCP | 7336 | 7336 | Automated testing |

---

## Step 3: Deploy the HMOPT Docker Environment

Follow the `Quick_Start_English.md` to set up the Docker container, index the kernel, and start MCP services.

### 3.1 Clone the project

```bash
git clone <repo_url> -b <branch_name>
cd hm-kernel-llm-opt
```

### 3.2 Configure environment variables

```bash
cp .env.docker.example .env.docker
vim .env.docker
```

Key variables to set:

```bash
# === Required paths ===
PROJECT_REPO_PATH=/path/to/your/project/trunk/
KERNEL_REPO_PATH=/path/to/your/kernel/source/

# === LLM API ===
HMOPT_LLM_BASE_URL=http://<your-llm-host>:<port>/v1
HMOPT_LLM_API_KEY=<your-api-key>
HMOPT_LLM_MODEL=glm-4.7
HMOPT_EMBEDDING_MODEL=qwen3-embedding-8b

# === MCP server API key (optional) ===
HMOPT_MCP_SERVER_API_KEY=<your-mcp-key>

# === Neo4j ===
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your-neo4j-password>

# === Path aliases (if compile_commands.json uses different host paths) ===
# HMOPT_PATH_ALIAS=/old/host/path:/new/host/path
```

### 3.3 Build and start the Docker container

**Option A — Build locally:**

```bash
bash scripts/docker_oneclick.sh up
```

**Option B — Use a prebuilt image bundle:**

```bash
# Copy the image bundle into dist/
scp <user>@<host>:<path>/hmopt_bundle.tar.gz dist/

# Load and start
bash scripts/docker_oneclick.sh load-images
bash scripts/docker_oneclick.sh up-prebuilt
```

### 3.4 Index the kernel source

If you have a `compile_commands.json` from a Yocto build, first generate it:

```bash
python scripts/parse_compilelog.py \
  -i <yocto_build_log> \
  --host-prefix <host_prefix> \
  --docker-trunk <docker_prefix>
```

Then run indexing:

```bash
bash scripts/docker_oneclick.sh index \
  --repo-path <kernel_source_path> \
  --compile-commands-dir <compile_commands_dir>
```

### 3.5 Start MCP services

Start the main MCP server (kernel index):

```bash
bash scripts/docker_oneclick.sh mcp
```

Or start all MCP services at once:

```bash
bash scripts/docker_oneclick.sh oneclick
```

Verify the MCP server is running:

```bash
curl -s http://127.0.0.1:7332/health
```

---

## Step 4: Deploy the OpenCode Multi-Agent Harness to the Kernel Directory

### 4.1 Copy the `.opencode/` directory to your kernel workspace

The `.opencode/` directory contains the full multi-agent harness — agents, pipelines, skills, memory, and state tracking. Copy it to the kernel directory where you will run OpenCode:

```bash
cp -r /path/to/hm-kernel-llm-opt/.opencode/ /path/to/your/kernel/source/.opencode/
```

This copies:

| Directory | Purpose |
|-----------|---------|
| `agents/` | Agent prompt definitions (starter, manager, researchers, reviewer, coder, tester) |
| `pipelines/` | Pipeline preset cards (generic, memmgr, hyperhold, workqueue, sync) |
| `skills/` | Reusable capability packs (instruction-count-first, research-discipline, language-config, etc.) |
| `docs/` | Living design documents and bootstrap context |
| `memory/` | Long-term memory across analysis runs |
| `state/` | Task state tracking |
| `plans/` | Approved optimization plans |
| `reviews/` | Plan and code review outputs |
| `bench/` | Validation reports |
| `config.yaml` | Workspace configuration (language setting, etc.) |

### 4.2 Configure the workspace language (optional)

Edit `.opencode/config.yaml` in your kernel directory to set the session language:

```yaml
# Set to zh-CN for Chinese or en for English
language: zh-CN
```

This controls the language of all agent dialogue, analysis, reviews, and documentation prose. Code, commit messages, and technical identifiers remain in English regardless.

### 4.3 Launch OpenCode in the kernel directory

```bash
cd /path/to/your/kernel/source
opencode
```

### 4.4 Start a multi-agent analysis session

**Option A — Interactive (paste the prompt):**

Stage a pipeline session first:

```bash
# From the hm-kernel-llm-opt project directory
python3 -m hmopt.cli start-pipeline \
  --profile generic_full \
  --target <target_path_or_subsystem>
```

This generates a staged prompt at `.opencode/state/current_prompt.md`. Copy and paste it into the OpenCode session.

**Option B — Use the wrapper script:**

```bash
bash /path/to/hm-kernel-llm-opt/scripts/run_opencode_pipeline.sh \
  --profile hyperhold_full \
  --target sysmgr/memmgr/mem/swap/hyperhold/hp_iotab.c \
  --start-mcp \
  --launch-opencode
```

**Option C — Manual prompt:**

In the OpenCode session, directly provide the task to the starter agent:

```
Profile: generic
Target: <your target file or subsystem>
Objective: reduce instruction count on hot path
Pipeline preset: generic_full
```

The `kernel-pipeline-starter` agent will load the pipeline, skill packs, and bootstrap docs, then hand off to the `os-opt-manager` for the full staged workflow.

### 4.5 Multi-agent workflow overview

Once the session starts, the pipeline follows this stage order:

```
kernel-pipeline-starter
  └─> os-opt-manager (routing)
        └─> research specialist (analysis + IC hypothesis)
              └─> kernel-plan-reviewer (approve / reject plan)
                    └─> kernel-code-agent (implement approved plan)
                          └─> kernel-code-reviewer (review code)
                                └─> kernel-tester-agent (conditional validation)
```

At each gate (plan review, code review), the workflow pauses for human approval. Handoff packets carry all context between agents.

Available pipeline presets:

| Preset | Use When |
|--------|----------|
| `generic_full` | Any kernel target (auto-routes to specialist) |
| `memmgr_reclaim_full` | Memory reclaim and allocator analysis |
| `hyperhold_full` | Swap I/O, compression, hpio, iotab |
| `workqueue_full` | Workqueue and thread-pool optimization |
| `sync_review` | Lock scope, race, and synchronization review |

---

## Troubleshooting

### MCP server not reachable

```bash
# Check container is running
docker ps | grep hmopt

# Check MCP port is exposed
curl -v http://127.0.0.1:7332/health

# Check container logs
bash scripts/docker_oneclick.sh logs
```

### Neo4j not starting

```bash
# Enter the container and check Neo4j status
bash scripts/docker_oneclick.sh shell
neo4j status
```

### Path alias issues

If indexed files cannot be found because `compile_commands.json` uses different host paths, set `HMOPT_PATH_ALIAS` in `.env.docker`:

```bash
HMOPT_PATH_ALIAS=/old/host/path:/current/host/path
```

Then restart:

```bash
bash scripts/docker_oneclick.sh down
bash scripts/docker_oneclick.sh up-prebuilt
```

### OpenCode cannot find `.opencode/` harness

Make sure you:
1. Copied `.opencode/` into the kernel directory (not a subdirectory).
2. Launched `opencode` from the kernel directory root.
3. The directory structure is `<kernel_root>/.opencode/agents/`, not `<kernel_root>/.opencode/.opencode/agents/`.
