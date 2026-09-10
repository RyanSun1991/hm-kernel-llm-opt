# Evolution V2：首次配置与日常操作

本文提供从空状态开始的操作顺序。字段和异常处理细节见 [完整操作指南](EVOLUTION_V2_OPERATIONS_CN.md)，方案与接入验收见 [实施计划](EVOLUTION_V2_IMPLEMENTATION_CN.md)。命令按 Windows PowerShell 编写；本地流程、模拟演示与真实 Agent 执行分别说明。

## 1. 先确定四个位置

| 配置 | 含义 | 示例 |
|---|---|---|
| 平台目录 | 本仓库，包含 Python 实现与 `.opencode` 角色 | `C:/Users/irtos/ryan/workspace/hm-kernel-llm-opt` |
| 目标仓库 | 待分析/优化的 Git checkout；进入实施前应使用可隔离的 checkout/worktree | `C:/work/kernel` |
| 记录工作区 | 保存已有 `.opencode/memory`、reviews、experiments 等内容的目录 | 平台目录或目标仓库，不必相同 |
| 状态目录 | 保存 SQLite、证据和派发结果 | `平台目录/data/evolution-kernel` |

CLI 和 Evolution MCP 必须指向**同一个绝对状态目录**，否则会出现 CLI 已确认但 Agent 查不到候选的情况。另需确定模块责任人的稳定 actor ID；这里的 ID 是本地身份约定，不是账号认证。

首次建议选一个模块、一个候选完成全过程。当前执行角色要求一次任务有明确的源码范围和独立评审；设备并行调度不由本地 discovery 提供。

## 2. 安装并定义命令简称

前置：Python 3.10+、Git。已有专用虚拟环境时复用它，跳过创建。

```powershell
$EvoHome = "C:/Users/irtos/ryan/workspace/hm-kernel-llm-opt"
Set-Location -LiteralPath $EvoHome
python -m venv .venv
$EvoPython = "$EvoHome/.venv/Scripts/python.exe"
& $EvoPython -m pip install -r "$EvoHome/requirements-evolution.txt"

$EvoGit = "git" # 不在 PATH 时替换为 git.exe 的绝对路径
$EvoState = "$EvoHome/data/evolution-kernel"
$EvoRepo = "C:/work/kernel" # 替换为实际目标仓库
$env:PYTHONPATH = "$EvoHome/src"
$env:PYTHONIOENCODING = "utf-8"

function Evo {
    & $EvoPython -m hmopt.evolution.cli --root $EvoState --git-bin $EvoGit @args
    if ($LASTEXITCODE -ne 0) { throw "Evolution command failed; inspect the message above." }
}

Evo init
Evo --help
```

`Evo` 只是本 PowerShell 会话中的简称，新终端需重新设置变量和函数。这里不需要完整安装旧分析栈，不读取旧的 trace preprocessing，也不要求模型、Neo4j 或设备在线。

先运行可检查的协议演示：

```powershell
$EvoDemo = Join-Path $env:TEMP ("hmopt-evolution-demo-" + [guid]::NewGuid().ToString("N"))
Evo demo-v2 --output $EvoDemo
Get-Content -LiteralPath "$EvoDemo/summary.json" -Raw
```

演示使用输出目录下自己的状态库和临时 Git 仓库，不使用真实目标状态库。它会产生四阶段任务包并执行 Python 功能断言；IC 数字是模拟值。预期 simulation=true、最终 code_approved、不能晋升技能。不要把演示报告改成 hardware=true 来充当真机结果。

## 3. 配置并运行第一次发现

复制 [发现配置样例](../configs/evolution/discovery.example.json) 到 `data/evolution-kernel/discovery.local.json`，填写实际值，例如：

```json
{
  "repo_path": "C:/work/kernel",
  "workspace": "C:/Users/irtos/ryan/workspace/hm-kernel-llm-opt",
  "repo_id": "kernel-main",
  "revision": "HEAD",
  "owners": {
    "kernel/**": "kernel-owner",
    "mm/**": "memory-owner"
  },
  "hotspots": [],
  "page_size": 200,
  "max_pages": 5,
  "source_page_size": 200,
  "source_page_bytes": 8388608,
  "max_files": 5000,
  "top_k": 20,
  "scheduling_budget_s": 120
}
```

`workspace` 只应指向存放与本目标相关记录的目录；没有记录时，省略 workspace/repo_id 也能只挖掘 Git。owners 的路径以目标仓库根为基准，使用 `/`；本地 actor 必须与归属表一致。热点可先为空；之后提供的热点必须绑定扫描的完整 revision，不能拿旧版本画像直接使用。

```powershell
$EvoConfig = "$EvoState/discovery.local.json"
Evo check-contract discovery $EvoConfig
$batch = Evo run-discovery $EvoConfig --actor analyst | ConvertFrom-Json
$batch.id
$batch.data.status
```

返回 partial 时，检查阶段进度，用原配置续跑：

```powershell
$batch = Evo run-discovery $EvoConfig --actor analyst --batch-id $batch.id | ConvertFrom-Json
Evo show $batch.id --kind batch
```

requires_attention 需要修复来源，requires_partition 需要调整输入/预算并建立新批次；running 的恢复先确认旧工作者停止，再使用 `--recover-running`。不要无条件循环重试所有状态。

**首次运行通常只得到草稿，没有候选。** 因为规则尚未激活。`awaiting_review` 表示发现批次结束，不代表所有候选都已审批或源码已全量扫描；请同时检查 scan.coverage。

## 4. 审阅并激活第一条真实规则

```powershell
Evo list --kind pattern --limit 100
Evo list --kind history --limit 100
Evo list --kind source --limit 100
Evo seeds
```

从实际历史或评审中选一条适用范围清楚的草稿，读取出处、修复前后差异和风险。12 个 seeds 只是研究题目，不是可直接导入/激活的规则。

导出选中的规则正文，注意 `show` 返回的是包装记录，`import-pattern` 要的是内部 Pattern：

```powershell
$EvoPatternKey = "替换为实际PATTERN_ID@1"
$patternRow = Evo show $EvoPatternKey --kind pattern | ConvertFrom-Json
$pattern = $patternRow.data.pattern
$pattern.version = $pattern.version + 1
$pattern.status = "draft"
$EvoPatternFile = "$EvoState/pattern.curated.json"
$pattern | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $EvoPatternFile -Encoding utf8
```

编辑 pattern.curated.json：写清 `problem/diagnosis/remedy`、目标文件 glob、字面匹配条件、指标、适用前提、风险和真实 source_ids。保留未证实前提；只有实际核实后才能将对应启发式假设改写为有证据的适用条件。修改内容使用新版本，不能覆盖旧版本。Matcher 当前是字面匹配，不能填写正则表达式并期望其按正则运行。

```powershell
$checked = Evo check-contract pattern $EvoPatternFile | ConvertFrom-Json
Evo import-pattern $EvoPatternFile
$EvoPatternKey = "$($checked.contract.pattern_id)@$($checked.contract.version)"
Evo activate-pattern $EvoPatternKey --actor curator --note "已核对历史修复、适用前提和反例，该规则可用于限定范围的候选筛选。"
```

激活不会自动重新运行已经完成的 batch。显式重新 scan，或创建新 discovery 批次：

```powershell
$cfg = Get-Content -LiteralPath $EvoConfig -Raw | ConvertFrom-Json
$cfg.owners | ConvertTo-Json | Set-Content -LiteralPath "$EvoState/owners.json" -Encoding utf8
Evo scan $EvoRepo --owners "$EvoState/owners.json" --top-k 10
Evo list --kind candidate
```

没有热点或前提仍不确定时，候选可进入 workbench，仍能由责任人确认后开展交互分析。首次使用不必为了 pipeline 标签而编造热点或删掉风险。

## 5. 责任人确认一个候选

批量方式：

```powershell
$sheet = Evo review-sheet --output "$EvoState/reviews" --owner kernel-owner --limit 10 | ConvertFrom-Json
$sheet.json_path
$sheet.markdown_path
```

读取 review.md，在 review.json 中只编辑每项 decision 和 note：confirm/reject/空白；作决定时理由至少 10 个字符。

```powershell
Evo apply-sheet $sheet.json_path --actor kernel-owner
```

单项方式：

```powershell
$EvoCandidate = "替换为实际CANDIDATE_ID"
$row = Evo show $EvoCandidate | ConvertFrom-Json
Evo decide $EvoCandidate confirm --actor kernel-owner --version $row.version --request-id "$EvoCandidate-owner-001" --note "确认该候选属于本模块，来源及适用条件已核对，可以开展方案设计。"
```

两种方式选一种即可。确认前目标 HEAD 必须仍是候选基线，且无已跟踪文件的未提交修改。owner 确认只允许进入方案阶段；方案通过后才能修改批准文件。

## 6. 选择执行方式

无论上一节采用批量还是单项确认，都先选定本次要执行的候选。批量确认不等于批量启动所有候选：

```powershell
$EvoCandidate = "替换为刚确认的实际CANDIDATE_ID"
$row = Evo show $EvoCandidate | ConvertFrom-Json
$row.data.stage # 首次派发前应为 confirmed
```

### A. 使用已有团队 Multi-Agent Harness

复用已经能正常调用模型和 `delegate` 的团队 OpenCode 环境。Evolution 模块本身不调用模型；`.opencode/config.yaml` 只控制语言，`configs/model_server.yaml` 不会自动配置 OpenCode 的 provider/model。

将 [Evolution MCP 样例](../examples/opencode.evolution.local.jsonc) 中的服务项合并到实际 OpenCode 项目配置 `opencode.jsonc`，保留原有模型、插件和其他 MCP 项。需替换：Python 路径、PYTHONPATH、Git 路径、状态目录。CLI 使用哪个 EvoState，MCP 的 --root 就填哪个绝对路径。该样例采用本仓库已有的 `mcp.<server>` 格式，与 [OpenCode 普通版 MCP 文档](https://opencode.ai/docs/mcp-servers/) 对齐；不要混用另一个 OpenCode V2 配置协议。

OpenCode 项目配置的位置与合并规则见 [官方配置文档](https://opencode.ai/docs/config/)。先在包含本仓库 `.opencode/agents` 的平台项目中启动，以便读取 manager 与角色文件。MCP 的绝对路径以 OpenCode 所在主机为准：在 Linux/容器运行时，将所有 Windows 路径换为容器可见路径，且目标 checkout、状态库和产物必须在执行端可见。

在团队环境检查：

```text
opencode --version
opencode mcp list
```

确认 hmopt_evolution 已连接，实际工具目录有 12 个 Evolution 工具，并可列出与 CLI 相同的候选。客户端可能为名称添加前缀，例如 `hmopt_evolution_evolution_show`；应按真实工具名调用。

已有平台工具按任务接入：研究角色需要已有的语义索引/Sequential Thinking 服务；实施需要访问隔离 checkout、Git 与适用构建工具；真机验证需要构建、设备/测试、原始 IC 或其他指标采集服务。Evolution MCP 不提供这些工具的实现。

派发已确认候选：

```powershell
$dispatch = Evo dispatch $EvoCandidate --output "$EvoState/dispatches" --actor coordinator --request-id "$EvoCandidate-architect-001" | ConvertFrom-Json
$dispatch.state_path
```

该命令只生成文件。然后在 OpenCode 选中 os-opt-manager，把返回的 state_path 作为实际任务路径，例如：

```text
请读取并按 .opencode/commands/evolve-candidate.md 执行 Evolution 任务。
任务文件：C:/实际目录/dispatch-...-confirmed-v2/task.json
先通过 hmopt_evolution 服务核对当前状态，再执行获准阶段。
每个角色都必须接收绝对任务路径、当前版本、冻结策略与独立产物目录。
```

若客户端已发现对应自定义命令，也可调用 `/evolve-candidate` 并附上绝对任务路径；不要保留模板占位符。自定义命令参数行为应按所用版本核对，参见 [官方命令文档](https://opencode.ai/docs/commands/)。

manager 按 architect→implementer→reviewer→validator 前进，每阶段提交成功后重新读取服务，并取得下一阶段的新任务包。缺少证据或服务拒绝时停止当前阶段，不能靠编辑 Markdown 状态继续。

**当前自动执行的前置缺口需要明确检查：** 仓库 manager 使用 `delegate: true`、`task: false`，但仓库不附带 delegate 插件/运行时实现。普通 OpenCode 的原生子代理接口是 Task，不能仅通过加一个 MCP 项保证现有团队 delegate 流程可用。应复用原有已落地运行时；若要改为原生 Task，需要单独适配、验证角色隔离和返回契约。[官方 Agent 文档](https://opencode.ai/docs/agents/)

现有专家/测试角色还包含旧 IC 默认目标、固定目录和设备流程。manager 的 Evolution 分支已要求传递冻结策略、绝对任务路径及权限，但实际 delegate 运行时必须把这些覆盖约束传给子角色并核验输出。尤其正确性任务不能被旧 tester 的 IC/刷机默认流程带偏。本机工具会话未在 PATH 发现 opencode，因此本轮没有验证真实 OpenCode 模型执行。

### B. 尚未接入 delegate 时，使用 CLI 分阶段提交

同一份门禁可由人或已有工具分别产出方案、代码、评审与测试，不依赖自动 manager。`Evo handoff $EvoCandidate` 返回当前允许角色与证据；按 [基础审批契约](EVOLUTION_QUICKSTART_CN.md) 生成文件。

| 动作 | payload 文件 | 必要内容 |
|---|---|---|
| approve_plan | plan-review.json | `{ "plan": 完整Plan, "review": 独立Review }` |
| record_implementation | implementation.json | `{ "revision": "完整实施提交ID" }` |
| approve_code | code-review.json | Review 对象本身，摘要绑定 implementation_digest |
| validate | ab-report.json 或 correctness-report.json | 冻结策略及真实采集结果 |

每个动作前先 show 获取最新版本。以方案评审为例：

```powershell
Evo schema plan
Evo schema review
Evo check-contract plan "$EvoState/plan.json"
$row = Evo show $EvoCandidate | ConvertFrom-Json
Evo decide $EvoCandidate approve_plan --actor plan-reviewer --version $row.version --request-id "$EvoCandidate-plan-review-001" --payload "$EvoState/plan-review.json"
```

独立 Review 的 subject_digest 使用 check-contract 返回的规范化摘要，不能使用原 JSON 文件字节摘要。后续登记实际实施提交，再独立评审真实差异。身份、准确文件范围、基线和评审摘要要求与 MCP 完全相同。

## 7. 方案中先冻结如何验收

| 任务 | Plan.validation | 必须准备 |
|---|---|---|
| 性能优化 | metrics、minimum_pairs、设备/工作负载/环境 | 主指标方向与阈值、护栏、基线/候选镜像和配对原始样本 |
| 正确性修复 | kind=correctness、required_checks、reproduction_checks、execution_kind | 基线缺陷复现、修改后必要检查、产物和日志证据 |

性能指标按已判定瓶颈选择，不默认所有任务都缩减 IC。正确性可选择经批准的 local 执行，具体策略见 [样例](../configs/evolution/correctness-policy.example.json)；样例零摘要是占位符，必须替换成实际配置/环境摘要。

已有 IC compare 数据在方案、实施和代码评审通过后转换：

```powershell
Evo schema ic-manifest
Evo convert-ic $EvoCandidate "$EvoState/compare.json" "$EvoState/manifest.json" --output "$EvoState/ab-report.json"
$row = Evo show $EvoCandidate | ConvertFrom-Json
Evo validate $EvoCandidate "$EvoState/ab-report.json" --actor validator --version $row.version --request-id "$EvoCandidate-validation-001"
```

正确性则将最后一步的文件换为通过 schema 校验的 correctness-report.json。IC 转换只支持冻结的单一指令数主指标；有时延/内存护栏时需完整多指标采集报告。设备 ID、镜像、配置摘要与样本必须来自采集流程；当前没有填一个设备地址就自动完成全部真机绑定的统一配置项。

## 8. 查看结果与沉淀

```powershell
Evo show $EvoCandidate
Evo audit $EvoCandidate
Evo quality --pattern-key $EvoPatternKey
Evo list --kind skill
Evo recall "目标模块 关键机制" --limit 3
```

真实验证通过后，选择该候选的 skill ID，由独立 curator 晋升：

```powershell
$EvoSkill = "替换为实际SKILL_ID"
$skill = Evo show $EvoSkill --kind skill | ConvertFrom-Json
Evo promote $EvoSkill staging --actor curator --version $skill.version --request-id "$EvoSkill-stage-001" --note "真实验证达到冻结要求，适用范围和独立评审已核对，纳入团队试用。"
```

需要审阅出口时 `Evo export-bundle --output 新目录 --actor curator`；目标必须不存在、父目录必须存在。导出的是脱敏评审包，本地 tier=hub 不代表团队 Skill Hub 已发布。结果为 inconclusive 时只能由 owner 按原策略批准重测；不要换阈值或镜像复用旧批准。

日常运行顺序是：新 discovery 批次→策展有价值新规则→显式 scan→owner 审阅→派发执行→查看原始验证与质量→独立策展。第一轮的实际人工操作集中在规则策展、候选确认、冻结验收方案和知识晋升。
