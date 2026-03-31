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

The `.opencode/commands/` directory contains pre-configured slash-command files. Each file is a complete prompt that references all necessary agents, pipelines, skills, memory, and config via `@<path>` annotations. OpenCode expands these references inline so the agent receives full context in one shot.

#### 4.4.1 Using a built-in command

1. Launch OpenCode in the kernel directory:

   ```bash
   cd /path/to/your/kernel/source
   opencode
   ```

2. Type `/` in the OpenCode session to see all available commands.

3. Select a command (e.g., `optimize_generic`) — the full prompt is injected automatically.

4. The `kernel-pipeline-starter` agent loads the pipeline, skill packs, bootstrap docs, and memory, then delegates to the `os-opt-manager` for the staged workflow.

#### 4.4.2 Available commands

| Command | Pipeline | Description |
|---------|----------|-------------|
| `/optimize_generic` | `generic_full` | Full pipeline for any kernel target with automatic specialist routing |
| `/optimize_memmgr_reclaim` | `memmgr_reclaim_full` | Memory reclaim and allocator-coupling deep analysis |
| `/optimize_hyperhold` | `hyperhold_full` | Swap I/O, compression, hpio, iotab, eid optimization |
| `/optimize_workqueue` | `workqueue_full` | Workqueue and thread-pool dispatch optimization |
| `/review_sync` | `sync_review` | Lock scope, race, and synchronization safety review (no implementation) |
| `/research_only` | `generic_full` | Research and analysis only (stops before implementation) |

#### 4.4.3 Customizing a command before use

Before triggering, open the command file and set your target:

```bash
vim .opencode/commands/optimize_generic.md
```

Change the `Target:` line:

```
Target: sysmgr/memmgr/mem/swap/hyperhold/hp_iotab.c
```

Optionally refine the `Objective:` to narrow the analysis scope. Save, then trigger via `/optimize_generic` in OpenCode.

#### 4.4.4 Creating your own command

Copy any existing command as a starting point:

```bash
cp .opencode/commands/optimize_generic.md .opencode/commands/my_custom_task.md
```

Edit the file to adjust:

- **Profile / Pipeline** — pick from `.opencode/pipelines/` or keep `generic_full` for auto-routing
- **Target** — the kernel file or subsystem path to analyze
- **Objective** — what the pipeline should achieve (research-only, optimization, review, etc.)
- **Skill packs** — add or remove `@.opencode/skills/*.md` references as needed
- **Bootstrap docs** — add subsystem-specific docs from `.opencode/docs/` if available
- **Memory packs** — add target-specific memory files from `.opencode/memory/` for context reuse

Then trigger in OpenCode by typing `/my_custom_task`.

#### 4.4.5 Command file anatomy

Here is the structure of a typical command file:

```markdown
@kernel-pipeline-starter @.opencode/agents/kernel-pipeline-starter.md

Profile: generic_full @.opencode/pipelines/generic_full.md
Target: sysmgr/pwrmgr
Objective: Analyze and optimize this target using the full generic pipeline
  with automatic routing, research, implementation, review, validation,
  and memory updates.

Skill packs:
- @.opencode/skills/instruction-count-first.md
- @.opencode/skills/research-discipline.md
- @.opencode/skills/optimization-funnel.md
- @.opencode/skills/handoff-contract.md
- @.opencode/skills/implementation-guardrails.md
- @.opencode/skills/validation-flight-check.md
- @.opencode/skills/memory-accumulation.md
- @.opencode/skills/language-config.md

Memory packs:
- @.opencode/memory/global_lessons.md

Bootstrap docs:
- @.opencode/docs/harness_engineer_system.md

Config:
- @.opencode/config.yaml
```

Key elements:

| Element | Purpose |
|---------|---------|
| `@kernel-pipeline-starter` | Tells OpenCode which agent to invoke |
| `@.opencode/agents/kernel-pipeline-starter.md` | Agent prompt (expanded inline) |
| `@.opencode/pipelines/generic_full.md` | Pipeline preset (stage order, load-first list) |
| `@.opencode/skills/*.md` | Skill packs (rules every agent must follow) |
| `@.opencode/memory/*.md` | Long-term memory (reused across runs) |
| `@.opencode/docs/*.md` | Bootstrap context and design notes |
| `@.opencode/config.yaml` | Workspace config (language, etc.) |

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
