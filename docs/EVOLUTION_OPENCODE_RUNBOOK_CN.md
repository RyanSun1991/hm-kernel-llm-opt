# Evolution on opencode：配置、使用与验收

> 当前配置统一复用 `HMOPT_MCP_CONFIG` 对应的平台 YAML。后台并发挖掘、多仓调度与验证适配器见[多仓指南](EVOLUTION_MULTI_REPO_RUNBOOK_CN.md)，审批与质量评测见[生产指南](EVOLUTION_PRODUCTION_RUNBOOK_CN.md)。历史分析协议见[历史挖掘设计](EVOLUTION_HISTORY_MINING_DESIGN_CN.md)，归并、复核与审批档案见 [Skill 工作流实施](EVOLUTION_SKILL_WORKFLOW_IMPLEMENTATION_CN.md)。

本指南对应主力 `opencode` 分支上的原生接入。设计见
[方案与实施计划](EVOLUTION_OPENCODE_DESIGN_CN.md)，现有平台见
[分支代码理解](OPENCODE_BRANCH_UNDERSTANDING_CN.md)。
**首次使用请从[三步快速开始](EVOLUTION_QUICKSTART_CN.md)进入：setup 在原 YAML 登记范围，doctor 检查，工作台触发。**
必要运行字段统一见[实际运行配置](../configs/evolution/README.md)。下文的 CLI 自动使用同一
`HMOPT_MCP_CONFIG`；任务 JSON 是运行产物，不需要另外创建状态目录或部署配置。

## 1. 安装与目录

下面使用 Linux/Bash 说明生产部署。Windows 使用同样的 `python -m` 命令，把路径换为本机绝对路径；
不要把 Bash 的变量赋值原样粘贴到 PowerShell。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,validation]'

python -m hmopt evolve doctor
python -m hmopt evolve --help
python -m hmopt resolve 'mm/vmscan.c::shrink_node' --stage research
```

validation extra 提供原始 xlsx 解析。需要建索引时另装 `.[indexing]`，按现有索引指南配置 clangd/SCIP、编译数据库与数据库。
Evolution 的本地挖掘、确认单和 schema 命令不依赖模型服务。
实际 Agent 运行还需要现有 OpenCode provider 和对应 MCP 配置。

## 2. 接入 OpenCode

Evolution 的 38 个工具已注册到 `src/hmopt/api/` 的主 MCP，与已有 5 个内核索引工具使用同一个入口。
多 Git 工作区使用新增的 `evolution_workspace`、`evolution_scan`、`evolution_experiment`，见 [多仓指南](EVOLUTION_MULTI_REPO_RUNBOOK_CN.md)。
首选 `setup` 在原平台 YAML 中生成 `evolution` 节，主 MCP 继续使用已有 `HMOPT_MCP_CONFIG`，重启后生效。
CLI 自动复用同一配置，也可用 `python -m hmopt evolve --config /absolute/configs/app.yaml ...` 显式选择。
无需再添加 `mcp.evolution` 连接，也不需要分别维护 state、workspaces、artifacts 与 profile 的多份配置。
主服务名称和已有 OpenCode 配置键保持兼容，仍可叫 `hmopt_kernel_index`。

本地 stdio 使用 `python -m hmopt.api.mcp_stdio`。
将 [示例](../examples/opencode.evolution.workbench.jsonc) 的 `environment` 合并到已有主 MCP 条目，
替换 Python 和目录路径；尚未配置主 MCP 时才添加示例中的整个条目。
保留已有 `HMOPT_MCP_CONFIG`、provider、`default_agent=assistant` 和其他 MCP 配置。
setup 保留原平台字段和 OpenCode 连接，仅追加业务范围并输出连接 fragment。

发现 profile 从 `evolution.source_workspace.selected` 自动生成；不需要维护独立的 profile JSON。
状态、任务与工件路径继承平台配置。旧 JSON 和 `HMOPT_EVOLUTION_*` 覆盖仍兼容，
`doctor` 在 `environment_overrides` 中列出旧变量；迁移时保留原归档路径和项目身份后再清除覆盖。
主服务默认注册 Evolution 工具；启动、列工具和查看 discovery profiles 不创建 Evolution 数据库，
首次需要读写 store 的工具调用才初始化该实例。索引检索仍使用现有 `HMOPT_MCP_CONFIG`。
源码路径须为 Git 仓库顶层。修改平台配置后重启 MCP；旧批次只接受原配置摘要，不能扩大旧批次范围。
远程容器返回的路径必须通过共享挂载在 OpenCode 进程侧可见；本版本不自动搬运远端文件。

使用远程主 MCP 时，保留已有 OpenCode `remote` 条目的 URL。在服务端维护原平台 YAML，
配置及业务目录均须在服务端可见；本地 setup 的产物不会自动上传或改写为容器路径。
然后沿用 `bash scripts/run_mcp_server.sh`（默认 7331 端口）或：

```bash
python -m uvicorn hmopt.api.mcp_server:app --host 127.0.0.1 --port 7331
```

OpenCode 连接同一 `/mcp` endpoint，Evolution 沿用该主服务的鉴权配置。
`run_all_mcp_servers.sh` 启动主 MCP 时也会带上这组工具，但仍不启动 Build/Git。
Docker Compose 的主服务环境变量配置见 `.env.docker.example`。
配置中的目录必须使用容器内可见路径；新增挂载需在部署配置中显式提供。

在 OpenCode 检查能列出 `evolution_show`、`evolution_digest`、`evolution_handoff`、
`evolution_dispatch`、`evolution_submit`、`evolution_validate` 等工具。
Kernel Index 与 Evolution 共用主 MCP；Team Memory、Build、Flash、AutoTest 等服务仍按原配置接入。

在真实 OpenCode 环境执行 `opencode mcp list`，确认已有主 MCP 已连接；然后在工作台请求列出
`evolution_discovery_profiles`。OpenCode 可能为工具名附加服务名前缀，应选择实际注册的工具。
项目提供的是可合并配置，仓库默认 `opencode.jsonc` 不会被自动改成当前机器的私有路径。
本次已验证 MCP stdio 协议；未声称启动了真实 OpenCode 模型会话。
配置字段、MCP 可见性及命令参数机制分别参考
[OpenCode MCP 文档](https://opencode.ai/docs/mcp-servers/) 和
[OpenCode Commands 文档](https://opencode.ai/docs/commands/)。
示例 `timeout` 用于获取工具列表，不能当成发现批次的执行时限。

**旧入口兼容。** `python -m hmopt.evolution.cli --root ... mcp-stdio` 仍可用于已有独立部署，
内部复用 `hmopt.api.evolution_mcp_service` 的注册逻辑。新接入使用上述统一入口；
从旧配置迁移后关闭旧 `mcp.evolution` 条目，避免同一工具组重复连接。
旧 CLI 的 `--root`、`--git-bin` 仍可显式覆盖；新接入无需重复指定。

### 2.1 工作台交互入口

```text
/evolve-discover kernel
/evolve-discover kernel <上次返回的完整批次ID>
/evolve-candidate <candidate-id> status
/evolve-candidate <candidate-id> full
/evolve-candidate <candidate-id> stage
/evolve-candidate <candidate-id> research
/evolve-candidate <candidate-id> plan
/evolve-candidate <candidate-id> implement
/evolve-candidate <candidate-id> review
/evolve-candidate <candidate-id> validate
```

`/evolve-discover` 由 researcher 调用有预算的发现操作，遇 awaiting_analysis 时用现有模型处理至多五个提交；partial 时显式续跑，
遇词典激活/责任人确认门暂停。`/evolve-candidate` 由 coordinator 调用现有独立角色。
`kernel` 是已登记项目名；使用其他项目或旧部署时替换为实际 profile 名。
`full` 从当前已许可阶段执行到验证结论；`stage` 只推进当前服务阶段一次；`status` 只读。
`research` 仅产研究材料；`plan` 包含研究、方案和独立方案评审；其余指定步骤必须满足已有前置状态，
不会为了“只跑验证”自动实施代码。只传 candidate ID 或旧版绝对 task.json 路径时默认 full。
scope 是工作台 recipe 的执行合同；服务端强制状态、证据与版本门禁，不将 scope 宣称为独立安全令牌。

工作台也可直接用自然语言请求：
“调用 Evolution 的 discovery_step，只对 kernel profile 执行 scan，展示覆盖率与候选，不进入实施。”
普通能力询问不触发操作；完整发现和候选执行是两个入口，中间的人工门不能自动跳过。

### 2.2 MCP 工具与可调用步骤

主 MCP 共 43 个工具，其中 Evolution 组 38 个。下表按职责组合列出原有 Evolution 工具；多仓调度、续扫和实验准备使用上面列出的三个新增工具，名称以服务内注册名称为准。

| 工作 | MCP 工具 | 边界 |
|---|---|---|
| 查看配置与记录 | `evolution_discovery_profiles`、`evolution_list`、`evolution_read`、`evolution_show` | profile 只由操作者注册，批次可按 ID 读取 |
| 完整发现 | `evolution_run_discovery(profile, actor, batch_id?)` | 历史挖掘→来源导入→蒸馏→筛选；终点为人工评审 |
| 单个发现步骤 | `evolution_discovery_step(profile, step, actor, cursor?, source_ids?)` | step 为 mine/sources/distill/scan；独立调用不更新批次进度 |
| 历史代码分析 | `evolution_history_analysis`、`evolution_submit_history_analysis` | 提供前后代码与输出 schema，接收模型分析；只生成经证据检查的 draft，不激活词典 |
| Skill 研究 | `evolution_code_context`、`evolution_prepare_research`、`evolution_submit_research` | 固定版本引用，模式归并/独立评审/候选适用性，保留方法快照 |
| 审批档案与队列 | `evolution_candidates`、`evolution_dossier` | 按 profile/owner/state 分页，追溯当时的审批版本、理由和证据 |
| 显式执行批次 | `evolution_create_batch`、`evolution_batch_next`、`evolution_batch_block` | 显式 ID 和范围、独立 worktree、占用与停止确认，未达真实验证门不能报完成 |
| 角色分发 | `evolution_handoff`、`evolution_dispatch`、`evolution_materialize_workspace` | 生成允许执行的包和工作区，MCP 本身不启动 Agent |
| 合同与评审 | `evolution_digest`、`evolution_submit` | 只提交方案评审、实现 revision、代码评审；不接受 owner confirm |
| 量化与验证 | `evolution_convert_ic`、`evolution_convert_lmbench`、`evolution_validate` | 原始证据转化、验证门分别执行；不接受 simulation 绕过 |
| 追踪与学习 | `evolution_evidence`、`evolution_audit`、`evolution_recall`、`evolution_capture`、`evolution_quality` | 捕获本地经验，不能自动发布 Skill Hub |

单步 `sources` 接受其返回的 cursor；`distill` 必须显式传入本 profile 的 source_ids，最多一个来源页。
`mine` 导入历史并建立待分析项；经模型分析和证据校验后才产生历史 pattern draft。`scan` 使用已经 active 的词典。
批次在 `awaiting_analysis` 等待代码分析结果；OpenCode 沿用现有模型执行分析后，可以继续同一批次。
`mine/scan` 每次分别解析注册 revision，需跨阶段固定同一个 commit 时使用发现批次。
`max_pages` 是每个阶段的页数上限，`scheduling_budget_s` 只控制阶段间调度，不抢占已运行的 Git 命令。
工具超时后先 list/read 批次，不能无 batch_id 盲目重新启动；running 的恢复仍需操作者 CLI。
已经 awaiting_review 的批次不因词典激活而重扫；用独立 scan 或新批次。
工具拒绝未声明的顶层参数，传 repo_path、owners 或 recover_running 不会静默忽略后开始工作。

### 2.3 专家审批与通知的当前边界

基础模式使用 `owners` 路径映射与 CLI/确认单；其中 `actor` 是可信本地标签，并非企业 SSO 身份。
可选生产模式已支持责任目录、SMTP TLS/webhook outbox、认证网关回调及原子审批归档，
在同一配置的 `production.approval` 中启用，详见 [生产使用指南](EVOLUTION_PRODUCTION_RUNBOOK_CN.md)。
企业网关负责实际登录认证和专家页面；邮件自由文本不构成批准。确认后状态成为 confirmed，
continuation 默认只提供工作台入口，仍由用户显式选择 full 或阶段范围，不自动唤醒执行会话。

## 3. 第一次 discovery

在工作台选择已登记项目；仓库和责任人来自同一平台 YAML：

```text
/evolve-discover kernel
# 继续未完成的同一批次，使用实际返回的完整 ID：
/evolve-discover kernel <batch-id>
```

此操作写入发现记录，不修改业务源码。多仓后台流程使用 `/evolve-workspace start` 和 `serve`，
见[多仓指南](EVOLUTION_MULTI_REPO_RUNBOOK_CN.md)。检查归档可直接使用：

```bash
python -m hmopt evolve list --kind batch
python -m hmopt evolve list --kind pattern --limit 100 --offset 0
```

检查结果的 status、coverage、source errors 和 cursor。`partial` 用返回的同一批次 ID 恢复；
`requires_attention` 先修复输入；`requires_partition` 需要分区，不能无限重试同一批。
首次 Git 导入从第一父链最早的一页开始，后续按 cursor 续页，直到目标 revision 的 caught_up=true。
合并提交使用相对第一父节点的 diff；不宣称分别挖掘了所有侧分支提交。单次分页不等于完整历史已处理，
单条大 diff 的 truncated 标记也应单独审查。

自动蒸馏只产生 draft。没有 active pattern 时零候选是正常结果，不能为了演示擅自激活未经核对的规则。
使用 `schema pattern` 看格式；完善来源、问题/诊断/修复、适用条件、反例、精确匹配范围后导入新版本，再显式激活：

```bash
python -m hmopt evolve import-pattern /path/to/reviewed-pattern.json
python -m hmopt evolve activate-pattern 'pattern-id@1' \
  --actor curator --note '已核对来源、适用条件、反例与匹配范围。'
```

激活后在工作台调用 `evolution_discovery_step(profile="kernel", step="scan", actor="operator")`；
多仓分区扫描使用 `/evolve-workspace scan kernel`。两者均复用已登记的仓库与 owners。
需要热点证据时更新对应项目的 `hotspots`，其 revision 必须匹配筛查目标。
`seeds` 输出研究模板，不能直接当作经过生产验证的 active pattern。

## 4. 纳入原生经验来源

仅在导入已有 Team Memory / Skill Hub 时，指定实际来源目录；它们是此次操作的输入：

```bash
python -m hmopt evolve import-native-memory /absolute/team-memory \
  --collection journal --contributor alice --project kernel --repo-id kernel --actor alice

python -m hmopt evolve import-native-memory /absolute/hm-skill-hub \
  --collection hub --repo-id kernel --actor curator

python -m hmopt evolve distill --actor curator
```

journal 必须明确 contributor/project；不会横向读取所有成员记录。Hub 只读 knowledge，
导入条目保留原生状态但不自动变为 active pattern。对分页结果继续传相同参数和 `--cursor`，
`has_more=false` 才表示该快照分页结束。

## 5. 责任人确认与启动

工作台展示确认单或发起已配置的专家审批。采用本地确认单时，由实际责任人执行：

```bash
python -m hmopt evolve review-sheet --output /absolute/review-sheets --owner alice
```

阅读返回目录中的 review.md，只编辑 review.json 每项的 `decision`（confirm/reject/空）与 `note`。
candidate/version/context 不能修改；每个决定需要具体理由。

```bash
python -m hmopt evolve apply-sheet /absolute/path/review.json --actor alice
```

确认后，直接使用已批准的候选 ID 启动工作台流程：

```text
/evolve-candidate <candidate-id> full
```

工作台负责生成 dispatch 和任务 workspace，使用平台默认目录。默认全候选模式按 recipe 连续取得下一阶段，实际过程是 researcher → architect → 独立 reviewer
→ implementer → 独立 reviewer → validator。每次服务晋级后产生新的 dispatch；失败会保留当前工件与原因。
普通工作台提问仍然走 assistant；发现候选不会自动改变其行为。

lmbench 任务须在方案评审前完成下一节的指标选择；将 profile 路径和规范摘要记录到任务 capsule，
architect 写入 Plan，独立 reviewer 核对后批准。已有 task.json 仍可作为旧入口传入。
同一 request_id 只能重放相同输入；capsule/execution.json 更新不等于服务批准。

## 6. 在方案评审前冻结 lmbench 指标

architect/validator 根据业务瓶颈选择主要评价指标及必要 guardrails，复用已有测试配方。
在任务 workspace 中保存指标 profile；它属于本任务的评审材料。单个指标的格式示意如下：

```json
{
  "kind": "lmbench_paired_suites",
  "metrics": [{
    "name": "syscall_latency",
    "system": "大核",
    "tool": "lmbench-lat",
    "metric": "lat_sys",
    "command": "lat_syscall null",
    "units": "microseconds"
  }]
}
```

字段必须对应真实 xlsx 行，指标不能统一替换为指令数。用现有接口查看协议、检查任务产物：

```bash
python -m hmopt evolve schema lmbench-profile
python -m hmopt evolve check-contract lmbench-profile /path/to/lmbench-profile.json
```

把输出 sha256 写进完整 Plan 的 `validation.measurement_profile_sha256`。
Plan 的 metrics 冻结 name、unit、direction、primary、min_improvement_pct、max_regression_pct；
另外冻结 minimum_pairs、device_id、workload_id、workload/environment hash。
独立 reviewer 用 `check-contract plan` 或 `evolution_digest(kind="plan")` 得到规范摘要。

绑定 profile 后，直接提交无来源的 ABReport 不能绕过原始转换门。
若需要改变 benchmark command 或指标身份，必须重新评审方案，不能在结果出来后挑选有利行。

## 7. 真机采集与正式验证

使用原有 AutoTest/Flash/Build 流程，分别采集明确 stock/feature 镜像的多组 suite。
每组保留 run_token、源码提交、镜像 hash、设备、环境、工作负载与 raw xlsx。
采集适配器根据真实运行生成 manifest，metrics 与批准的 profile 保持一致。
最少三组独立 suite pairs；同一 workbook 的 valueN 先求 suite 均值，不冒充跨镜像独立样本。

已接入[业务验证适配器](EVOLUTION_MULTI_REPO_RUNBOOK_CN.md#4-接入已有构建和真机脚本)时，
适配器返回 `{"lmbench_manifest": ...}`，框架完成原始转换和验证，无需手工维护 manifest 配置。
尚未接入时，由 validator 整理真实采集记录，按 `schema lmbench-manifest` 输出的协议导入。
schema 只定义结构，不能用占位摘要代替真实结果。手动导入步骤如下，路径均指向本次任务产物：

```bash
python -m hmopt evolve schema lmbench-manifest
python -m hmopt evolve convert-lmbench <candidate-id> \
  /path/to/lmbench-manifest.json --artifacts-root /absolute/collected-results \
  --output /path/to/ab-report.json
python -m hmopt evolve show <candidate-id>
python -m hmopt evolve validate <candidate-id> /path/to/ab-report.json \
  --actor validator-alice --version CURRENT_VERSION --request-id validation-attempt-1
```

raw 转换只规范化并存证，最后一个 validate 才判定 pass/fail/inconclusive。
缺原始列、重复 token/文件、环境/版本不一致、profile 不一致都应返回明确错误。
IC 使用 `convert-ic`；correctness 使用 `schema correctness-policy` 与 `schema correctness-report`，
要求基线复现与候选检查通过，不套用收益百分比。
correctness-policy 放在 Plan 的 `validation` 中，随方案冻结；无需维护单独的全局验证配置文件。

## 8. 沉淀进现有 Team Memory / Skill Hub

```bash
python -m hmopt evolve list --kind skill
python -m hmopt evolve quality
```

选择真实验证通过的本条执行经验，用独立 curator 把内部成熟度推进 staging：

```bash
python -m hmopt evolve promote SKILL_ID staging \
  --actor curator --version SKILL_VERSION --request-id curate-this-evidence \
  --note '已独立核对方案、实现、验证及适用范围。'
```

准备一个经过人工审阅、可共享的短内容 JSON：

```json
{
  "title": "已验证的方法及适用条件",
  "body": "十行以内说明做了什么、为何有效、证据与限制。",
  "applies_when": ["方法成立所需的具体条件"],
  "invalidated_by": ["出现什么条件时不能照搬"],
  "target_slug": "example-target"
}
```

两个 root 需已存在且不重叠：

```bash
python -m hmopt evolve export-native-memory SKILL_ID \
  --content /path/to/shareable-fact.json --memory-root /absolute/team-memory --hub-root /absolute/hm-skill-hub \
  --contributor alice --project kernel --actor curator --request-id native-export-1
```

返回 journal_id、journal_path、staging_path 与来源/文件哈希；状态仍是 `not_published`。
原生记录中的 `evolution:evidence:sha256-b32:...` 是对已核验摘要的可逆编码，避免被现有 Hub 的
长十六进制密钥检测误判；原始内容仍执行既有脱敏规则。用下面命令回查本实例证据：

```bash
python -m hmopt evolve resolve-native-evidence 'evolution:evidence:sha256-b32:...'
```

此操作只写所选条目，未重扫个人 journal，也未直接改 knowledge/skills。
继续按现有 [Skill Hub Runbook](Skill_Hub_Runbook_CN.md) 做 central curate、评审、CI、评测与 release。
内部 tier=hub 不等于完成这一发布过程。

## 9. 故障与验收边界

| 现象 | 处理 |
|---|---|
| 无候选 | 先查 active patterns、coverage 和 hotspot revision；不要伪造演示候选 |
| stale version / request conflict | 重读 service state；不修改旧批准上下文，不盲目重试 |
| workspace 不可见 | 核对 OpenCode 当前工程和服务端共享挂载，使用 materialize 返回路径 |
| 缺索引 | researcher 记录缺口并用冻结源码做可追溯复核；不能空口声称语义证明 |
| raw 转换拒绝 | 检查 manifest/profile、文件 hash、xlsx 原始列和 suite pair 身份 |
| 结果不确定 | 增加合规独立采样或调整另一次经评审实验，不强行标 pass |
| native 导出拒绝 | 检查真实通过、curator 独立、来源完整、内容/适用条件与原生 schema |

## 10. 本次验证记录（2026-09-10）

- 全量 `tests + hm-skill-hub/tools/tests`：971 passed、1 个旧 IC 证据精确相等断言失败、4 skipped、2 deselected。
  该断言已跟随新增 `kind=ic_compare` 来源契约更新；接口、CLI、轻量入口与端到端相关 **57 项复跑全部通过**，
  其中包含 3 项新增输入类型检查。没有剩余已知失败。
- 4 项 skip：本机未安装可选 LlamaIndex 栈的两个模块、无法创建符号链接的两项测试。
  排除的两项 Windows relay 测试分别涉及环境相关命令缺失假设和外部进程超时。
- Evolution Python/测试 Ruff 检查与格式检查通过；技能 registry 31 个条目和新 skill 格式校验通过。
- CLI/MCP 实际接口探针：profile规范哈希 → 冻结方案 → 原始xlsx转换 → MCP验证 → native journal/staging，全部跑通。
  使用离线夹具与声明身份，没有启动模型或设备，也没有发布真实 Hub 知识。
- 本次工作台交互补充：新增 4 个 MCP 工具后，共 19 个工具；CLI/实际 stdio 接口回归 **44 项通过**，
  覆盖配置冻结、分页恢复、四类独立步骤、路径/参数边界和人工门；命令合同/golden **9 项通过**；
  discovery、workspace workflow 与轻量入口另有 **48 项通过**，本次相关验证合计 **101 项通过**。
- 统一入口迁移：19 个 Evolution 工具合并到 API 主服务后，与 5 个索引工具共存。
  旧接口、轻量入口及索引渲染回归 **82 项通过**；统一 HTTP/stdio 协议测试 **13 项通过**，
  包括仅使用 setup 生成的单份配置启动真实 stdio MCP、完成发现批次并由兼容入口读取同一状态库。
- 单文件快速接入：setup 测试 **18 项通过、1 项 Windows 符号链接测试跳过**；
  doctor 最新回归 **11 项通过**；配置加载、CLI 与 MCP 实例一致性相关测试 **11 项通过**。
  上述统一入口与快速接入相关用例最终合并运行：**135 项通过、1 项跳过**。
  setup/doctor 默认输出中文简报，命令末尾 `--json` 提供机器输出；这些检查不运行模型或设备，也不代表生产收益。
- 当前仓库 `opencode@c50d26e` 只读 discovery：5 页累计133个第一父链提交、6个允许的工作区来源，
  零输入错误，终态 awaiting_review。未激活pattern，因此源码初筛明确返回 no_active_patterns，候选为零。

离线回归不等于已完成生产内核优化或真机收益验证。
现有角色共享 `.opencode/local/**` 的可写上限；分角色子目录是协作契约，不是操作系统级隔离。

生产后台挖掘、专家通知与认证回调、质量评测的配置及使用见 [生产增量指南](EVOLUTION_PRODUCTION_RUNBOOK_CN.md)。
