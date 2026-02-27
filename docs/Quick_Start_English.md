# HMOPT Docker Quick Start (English)

This document provides a simple end-to-end workflow for running HMOPT with Docker, indexing kernel code, and connecting OpenCode.

## 1) Project and Docker Main Functions

The project provides:

- **Kernel code indexing** with compile database support (`compile_commands.json`).
- **Embedded Neo4j** in Docker for graph/vector storage used by retrieval.
- **MCP server** for remote tool-based code retrieval.
- **One-click scripts** for build/start/load/index/service operations.

The main script is:

```bash
bash scripts/docker_oneclick.sh <action>
```

---

## 2) Setup / Deployment

### 2.1 Clone source code

```bash
git clone <repo_url> -b <branch_name>
cd hm-kernel-llm-opt
```

### 2.2 Run Docker

#### Option A: Build image locally

```bash
bash scripts/docker_oneclick.sh up
```

#### Option B: Load prebuilt image bundle

Copy the image package (`.tar.gz`) into this project directory first, for example via `scp`:

```bash
scp <user>@<host>:<path>/hmopt_bundle.tar.gz dist/
```

Then load and start:

```bash
bash scripts/docker_oneclick.sh load-images
bash scripts/docker_oneclick.sh up-prebuilt
```

---

## 3) Prepare and Index Kernel Code

### 3.1 Generate `compile_commands.json` from Yocto compile log

```bash
python ~/work/hm-kernel-llm-opt-main/scripts/parse_compilelog.py \
  -i <yocto_build_log> \
  --host-prefix <host_prefix> \
  --docker-trunk <docker_prefix>
```

Example:

```bash
python ~/work/hm-kernel-llm-opt-main/scripts/parse_compilelog.py \
  -i ~/code/scratch/tongkun/hione/work/trunk_new/kernel/hongmeng/build_tools/yocto/ng/build/tmp/work/aarch64-euler-elf/hm-sysmgr-nashvilleoh/git-r0/temp/log.do_compile.230241 \
  --host-prefix /home/ryan/code/scratch/tongkun/hione/ \
  --docker-trunk /work/trunk_new
```

### 3.2 Run kernel indexing

```bash
bash scripts/docker_oneclick.sh index --repo-path <repo_path> --compile-commands-dir <compile_commands_dir>
```

Example:

```bash
bash scripts/docker_oneclick.sh index \
  --repo-path /home/ryan/code/scratch/tongkun/hione/work/trunk_new/kernel/hongmeng/hm-verif-kernel/uapps/tppmgr \
  --compile-commands-dir /home/ryan/code/scratch/tongkun/hione/work/trunk_new/kernel/hongmeng/hm-verif-kernel/uapps/tppmgr
```


### 3.3 If `compile_commands.json` paths are from another host

If `compile_commands.json` points to old host paths (for example `/home/ryan/...`) but your current machine uses a different prefix (`/home/levi/...`), set `HMOPT_PATH_ALIAS` in `.env.docker`:

```bash
HMOPT_PATH_ALIAS=/home/ryan/mmrootdir/open_source:/home/levi/mmrootdir/open_source
```

You can provide multiple mappings with commas:

```bash
HMOPT_PATH_ALIAS=/old/prefix1:/new/prefix1,/old/prefix2:/new/prefix2
```

Then restart container/services so the new env takes effect:

```bash
bash scripts/docker_oneclick.sh down
bash scripts/docker_oneclick.sh up-prebuilt
```

---

## 4) Start MCP Server

```bash
bash scripts/docker_oneclick.sh mcp
```

By default in this repo, MCP is exposed on host port `7332` (container `7331`).

---

## 5) Configure OpenCode and Start OpenCode

Edit OpenCode config:

```bash
vim ~/.config/opencode/opencode.json
```

Use a config similar to:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-provider": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "local-provider",
      "options": {
        "baseURL": "http://10.90.56.33:20010/v1/",
        "apiKey": "<your key>"
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
        "Authorization": "<your key>"
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

Then start OpenCode as usual in your environment.

---

## Notes

- If you use the prebuilt-image flow, ensure `bash scripts/docker_oneclick.sh load-images` succeeds before `up-prebuilt`.
- Ensure your `repo-path` and `compile-commands-dir` are accessible from Docker runtime.
- If MCP access is blocked by host-header checks in your environment, configure MCP host settings in `.env.docker` as needed.
