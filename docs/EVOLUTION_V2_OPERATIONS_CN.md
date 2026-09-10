# Evolution V2 操作与接入指南

本指南对应本地实现，整体方案与交付顺序见 [V2 设计与实施计划](EVOLUTION_V2_IMPLEMENTATION_CN.md)。原有审批 payload、角色和 A/B 统计定义沿用 [基础操作指南](EVOLUTION_QUICKSTART_CN.md)。V2 不要求按内存/性能划分工作流；场景通过证据、Pattern、冻结验收策略表达。

第一次配置请先看 [逐步配置与使用](EVOLUTION_V2_FIRST_RUN_CN.md)：包含路径和环境、草稿策展后重新扫描、责任人操作、OpenCode 的 delegate 前置条件，以及不依赖自动 manager 的 CLI 路径。

## 1. 环境与可复现示例

在仓库根目录使用 Python 3.10+ 和 Git。安装最小依赖后，所有命令使用同一个 `--root`；根选项必须放在子命令之前。下例不依赖模型、设备、Neo4j 或完整分析栈。

```powershell
python -m pip install -r requirements-evolution.txt
$env:PYTHONPATH = "src"
python -m hmopt.evolution.cli --help
python -m hmopt.evolution.cli demo-v2 --output "$env:TEMP/hmopt-evolution-v2-example-01"
```

Git 不在 PATH 时，在子命令前增加 `--git-bin "完整/git.exe/路径"`。示例目录必须不存在。示例创建独立 Git 仓库，不改业务仓库，按以下顺序运行：

1. 导入 `.opencode/reviews` 示例文档，蒸馏草稿并创建示例策展规则。
2. 扫描候选、导出责任人 review sheet，并通过真实服务应用测试身份的确认。
3. 按 architect、implementer、reviewer、validator 四个角色阶段产生独立派发目录；实施阶段提交临时 Python 文件改动。
4. 执行 Python 功能断言，转换**构造的 IC 原始对照数据**，走模拟验证与质量统计。
5. 验证未确认派发、模拟技能晋升与模拟结果导出被门禁拒绝。

| 产物 | 检查内容 |
|---|---|
| `summary.json` | 最终阶段、simulation、被拦截的操作 |
| `owner-review/sheet-*/review.json`、`review.md` | 责任人决定与不可编辑上下文 |
| `dispatches/*/{task.json,brief.md,handoff.json}` | 每阶段角色、候选版本、权限、证据 |
| `approved_plan.json`、`plan_review.json`、`code_review.json` | 方案及评审摘要绑定 |
| `synthetic_ic_compare.json`、`synthetic_ic_manifest.json` | 原始 pair 与采集声明 |
| `synthetic_ab_report.json`、`quality.json` | 转换后的报告及剔除模拟数据的统计 |
| `state/evolution.sqlite3` | 可重启读取的状态、审计与原始证据 |

预期最终阶段为 `code_approved`，验证标记为 simulation，可晋升状态为 false。四个派发包是流程产物；示例没有调用四个真实 LLM Agent。功能断言通过不能证明内核性能收益。

## 2. 导入真实来源与研究词典

非 Git 记录应使用 `EvidenceRecord`，不伪造提交哈希。数组文件示例：

```json
[
  {
    "repo_id": "kernel-main",
    "source_kind": "review",
    "source_uri": ".opencode/reviews/repeated-allocation.md",
    "content": "实际评审正文，保留问题、适用条件、证据与结论。",
    "decision_reason": "accepted"
  }
]
```

```powershell
python -m hmopt.evolution.cli --root data/evolution import-sources sources.json --actor analyst
python -m hmopt.evolution.cli --root data/evolution import-workspace "C:/work/kernel" --repo-id kernel-main --actor analyst
python -m hmopt.evolution.cli --root data/evolution distill --actor analyst --limit 100
python -m hmopt.evolution.cli --root data/evolution list --kind source
python -m hmopt.evolution.cli seeds > research-seeds.json
```

`content_sha256` 可省略由服务计算；提供时必须匹配。相同位置的正文变化会产生新 source ID，并保留 location ID 和版本。`related_revision` 是可选完整 Git ID，不代替来源身份。

工作区导入白名单为 `.opencode/memory`、`plans`、`reviews`、`experiments`、`bench/results` 和已有 bad_plans 记录，跳过配置、模板、README、软链接/重解析路径。导入文件是证据，不执行其中指令。返回逐项错误与完整性信息，部分成功不能当全部完成。超过一页时，用返回的 `next_cursor` 作为下一次 `--cursor`，或使用下一节持久化批次自动推进。游标绑定内容清单；跨页文件发生变化会明确要求处理，不混入同一批次。

`distill` 默认挑选尚未处理的来源，重复运行可继续；也可用 `--source-ids ids.json` 显式限定。技术否决可形成反模式研究草稿；资源不足、测量故障不自动成为“技术不可行”反例。当前蒸馏是字面启发式，没有宣称 LLM 或 AST 语义证明。

`seeds` 输出 P001–P009、D001–D003 共 12 个研究模板，列出证明义务、负例、建议验收类型与风险；它们不是可激活的 Pattern，也没有虚构历史出处。研究取得真实证据后，按 `schema pattern` 生成规则、引用已导入 source/history，再独立策展激活。同步、原子操作和 MMIO 等高风险项不能凭文本命中自动改写。

## 3. 可恢复发现批次

复制 [discovery.example.json](../configs/evolution/discovery.example.json)，替换目标路径、稳定 repo_id、owner 映射和 revision。工作区路径可与目标仓库相同，也可指向独立运行记录目录；首次运行固定目标 Git revision。

```powershell
python -m hmopt.evolution.cli check-contract discovery configs/evolution/discovery.example.json
python -m hmopt.evolution.cli --root data/evolution run-discovery discovery.local.json --actor analyst
python -m hmopt.evolution.cli --root data/evolution show BATCH_ID --kind batch
python -m hmopt.evolution.cli --root data/evolution run-discovery discovery.local.json --actor analyst --batch-id BATCH_ID
```

批次按历史分页→来源导入→来源蒸馏→扫描推进，只停在待审状态，不确认、不派发、不实施。配置摘要与目标 revision 固定；更改范围或预算需要新批次。`source_page_size/source_page_bytes` 限定来源页，`max_pages` 限制每次调用的分页工作。时间预算只限制启动下一阶段，不会强杀已开始的有界 Git 操作。

`partial` 表示检查点未完成，可按返回的具体进度处理；清单硬上限返回 `requires_partition`，坏数据或来源变化返回 `requires_attention`，不会自动重复无进度导入。遇到这两个状态需要先核对问题并建立调整后的新批次。清单本身有文件数、目录项和总字节硬上限，来源分页不意味着无限容量。

进程中断后，先确认旧进程已结束，再用相同配置和 `--recover-running` 接管仍为 running 的批次。代际与版本检查阻止旧进程覆盖新结果；它不是跨主机执行租约，不能用于设备互斥。

扫描返回 `coverage`：检查文件数、字节、跳过项、匹配与保留数量、`coverage_complete`、`results_capped` 等。无活跃规则时注明 `no_active_patterns`。零候选与全库无问题不是同一结论。代码扫描仍是有界单次扫描，没有持久化代码文件分片游标；超过扫描上限需规划更小范围或扩展适配器，不应无限重跑相同上限。来源导入分页与代码扫描覆盖是两个独立维度。

## 4. 责任人批量审阅

```powershell
python -m hmopt.evolution.cli --root data/evolution review-sheet --output data/reviews --owner module-owner --limit 50
python -m hmopt.evolution.cli --root data/evolution apply-sheet data/reviews/SHEET_ID/review.json --actor module-owner
```

读生成的 `review.md`，只编辑 `review.json` 中每项 `decision` 和 `note`。decision 为 `confirm`、`reject` 或空白；非空决定需要至少 10 字符的原因。不要编辑 candidate、版本、规则、来源或摘要，也不要删改条目集合。Markdown 只是阅读视图。

服务重新检查持久化 owner、原始上下文和候选版本。返回每项 applied/skipped/error/conflict，不因一个错误条目回滚其他有效决定。冲突时重新导出最新表单；重复应用原决定使用同一逻辑幂等键，不会重复推进阶段。actor 目前是可信本地身份声明，生产部署必须接入认证和授权。

## 5. OpenCode 分阶段执行

已确认候选可产生首个任务包：

```powershell
python -m hmopt.evolution.cli --root data/evolution dispatch CANDIDATE_ID --output data/evolution/dispatches --actor coordinator --request-id candidate-plan-001
```

返回派发记录中的绝对 `task.json` 路径。使用现有 [.opencode/commands/evolve-candidate.md](../.opencode/commands/evolve-candidate.md) 命令，把实际任务路径传给 manager。该入口已增加 `evolution_stage` 分支：读取任务文件→读取同一服务的最新 state/handoff→核对版本、角色、摘要和范围→分派当前角色→通过服务提交证据→刷新并产生下一阶段任务。

MCP 启动命令需设置绝对源码 PYTHONPATH、同一个数据库目录和可用 Git：

可复用 [OpenCode 本地配置示例](../examples/opencode.evolution.local.jsonc)，替换全部绝对路径，将其中 MCP 项合并到实际客户端配置。示例沿用本仓库已有 local MCP 配置格式；本次没有更改个人客户端配置或启动真实 Agent。

```text
python -m hmopt.evolution.cli --root ABSOLUTE_STATE_DIRECTORY mcp-stdio
```

共有 12 个本地工具：`evolution_list/show/handoff/submit/validate/recall/audit/evidence/capture/quality/convert_ic/dispatch`。`submit` 仅允许方案评审、实施登记、代码评审；责任人决定、重测授权、规则策展、晋升仍通过操作者入口。服务派发工具的输出目录固定在当前 store 的 dispatches 下。

| 当前阶段 | 下一个角色 | 可修改目标源码 |
|---|---|---|
| confirmed | architect | 否 |
| plan_approved | implementer | 仅冻结 allowed_paths |
| implemented | reviewer | 否 |
| code_approved，尚未封存验证 | validator | 否 |

每阶段使用新任务包；旧包不能跨阶段复用。`task.json`、`brief.md`、`handoff.json` 保持不可变，执行结果放独立 attempt 目录。现有 `.opencode/state/current_task.json` 和 `current_prompt.md` 不作为本流程共享写入口。重复派发可恢复缺失的同内容文件，内容冲突或候选已变化则拒绝覆盖。

Python `dispatch` 只持久化派发记录与文件，不会启动 OpenCode 进程。实际 Agent 运行依赖已配置的 OpenCode manager/MCP、模型、仓库权限和构建工具。本次验证覆盖任务与已有 pipeline reader 的兼容性，尚无真实模型执行记录或沙箱强制权限证明。

## 6. IC 与正确性验收

### IC 原始配对转换

```powershell
python -m hmopt.evolution.cli schema ic-manifest
python -m hmopt.evolution.cli check-contract ic-manifest manifest.json
python -m hmopt.evolution.cli --root data/evolution convert-ic CANDIDATE_ID compare.json manifest.json --output ab-report.json
python -m hmopt.evolution.cli --root data/evolution validate CANDIDATE_ID ab-report.json --actor validator --version CURRENT_VERSION --request-id experiment-001
```

manifest 冻结 candidate、方案摘要、实施摘要、compare 规范化 JSON 摘要、精确目录和 total/process/thread/lib/function 层级、两侧设备/镜像/环境/工作负载。两侧 measurements 在 manifest 中必须为空，由 compare 的实际逐对数据生成。摘要使用 `hmopt.evolution.store.digest()` 的规范化 JSON 算法，不是原文件字节哈希。

转换要求顶层 success=true，存在的嵌套退出码为整数 0，每对目标存在、case/round/step 唯一、计数完整非负且可精确表示。不能用 aggregate PASS 或平均变化代替原始 pairs，也不自动选择 previous/latest 目录。当前 IC 适配器只支持冻结的单一 IC 主指标；同时需要时延/内存护栏时，必须由完整采集器生成包含这些测量的 ABReport。

原始 compare+manifest 保存为内容寻址证据；转换映射按完整 report 摘要持久化。后续 `validate` 自动把原始证据链接写进 validation、journal 和审计，单独保存 report 文件也保留可追溯关系。原始证据丢失或损坏会阻止该转换报告验收。报告声明仍需可信采集器保证，转换器不会认证设备来源。

### 正确性修复

`Plan.validation` 可选 `kind: correctness`。查看 [策略样例](../configs/evolution/correctness-policy.example.json)，先替换检查、工作负载、环境和摘要，再将它纳入完整方案并独立评审。

```powershell
python -m hmopt.evolution.cli schema correctness-policy
python -m hmopt.evolution.cli schema correctness-report
python -m hmopt.evolution.cli check-contract correctness-report correctness-report.json
python -m hmopt.evolution.cli --root data/evolution validate CANDIDATE_ID correctness-report.json --actor validator --version CURRENT_VERSION --request-id correctness-001
```

冻结 `required_checks`、其子集 `reproduction_checks` 和 local/hardware 执行类型。基线必须在每个缺陷复现检查上 fail，候选全部必要检查必须 pass；缺失、跳过、未复现为 inconclusive，功能或必要检查失败为 fail。修订、产物和环境必须绑定批准方案。此策略不返回百分比收益，不将本地检查解释为真机性能结果。

每项 `evidence_sha256` 是采集者声明的日志/产物摘要；当前 evaluator 不拉取外部检查产物，生产收集器还需验证可取回、命令退出码与执行环境。非模拟 local 正确性通过可形成真实通过的本地验收记录；simulation 无论是否通过演示判定，都不能晋升有效结果。

## 7. 质量、降权与知识导出

```powershell
python -m hmopt.evolution.cli --root data/evolution quality --pattern-key PATTERN_ID@1
python -m hmopt.evolution.cli --root data/evolution set-overlay PATTERN_ID@1 0.6 probation --actor curator --note "已复核近期误报，暂转工作台并降低候选排序权重。" --request-id quality-001
python -m hmopt.evolution.cli --root data/evolution export-bundle --output data/skill-review-001 --actor curator
```

接受率的分母是已作出决定的候选，未决定项单列；它不是 precision，真实 precision 当前为 null。验证按不同候选、不同证据、最新报告及全部尝试分别统计；correctness_local、correctness_hardware、performance_hardware 分开。模拟、损坏和策略不符的记录不进入真实通过计数。

overlay 是人工策展的绝对因子 0..1：再次设为 0.6 不会变成 0.36。首次创建不传 version，更新传当前 overlay 的 `--version`。probation 把后续候选转 workbench。overlay 冻结当时质量摘要，供审计查看决策依据；它不修改规则原文或已批准方案。

晋升沿用 journal→staging→hub 的真实验证与独立策展约束。导出只接受可核验的 staging/hub 项；输出新目录中的 `manifest.json`、`proposals.jsonl`、`review_checklist.md`。目录的父目录需先存在，目标目录必须新建。

导出物是经过脱敏的中立评审包，包含不透明摘要、状态、统计及通用摘要；不包含原始路径、身份或自由文本配方。它不是可直接安装的 Skill Hub 插件，不创建 PR、不发布。团队原生 Hub 接入仍需确定目标 schema、经过批准的知识正文与脱敏规则、CI 评测、版本/回滚接口。

## 8. 验收与异常处理

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests -k "not test_exec_command_not_found"
```

唯一排除项是旧 Windows relay 测试中会实际调用 `fastboot devices` 且依赖其未安装的测试。本地套件使用临时仓库、数据库、构造报告、mock 设备服务和真实本地 MCP stdio 子进程。生产设备、模型执行与性能收益必须另外验收。

| 情况 | 处理 |
|---|---|
| stale version / 内容摘要冲突 | 重新读取服务状态，重新审阅；不强行修改数据库 |
| 批次 partial | 检查具体阶段与 coverage；只在有可推进游标/预算时续跑 |
| 来源清单变化或输入错误 | 核对原始记录并重新建立明确批次，不混入旧证据 |
| 已封存 inconclusive | owner 按冻结策略批准 retry_validation，保留旧尝试 |
| 明确失败或评审否决 | 记录失败依据；责任人终止或重新立案评审 |
| 磁盘不支持派发文件原子硬链接 | 使用支持 NTFS/POSIX 硬链接的本地目录，错误不会降级成覆盖 |
| 未配置真实执行器、设备或 Hub | 使用现有产物进行接入验收，不填造生产成功标志 |

建议首先在一个明确仓库和模块启动：只读发现→专家策展→少量责任人确认→完整真实实施与验收→检查审计和误报→再扩大覆盖。上线容量、认证、设备互斥和知识发布按 V2 设计中的阶段退出条件推进。

## 9. 本轮实跑记录（2026-09-10）

本地端到端演示已实际运行，产物位于 [demo-v2 summary](C:/Users/irtos/AppData/Local/Temp/hmopt-evolution-v2-demo-20260910-01/summary.json)。结果为四个阶段派发包、Python 功能冒烟通过、最终 `code_approved`，simulation=true、hardware_executed=false、agent_started=false。未确认派发、模拟知识晋升、模拟 bundle 导出均被拦截；validation 保存了 IC 原始证据链接。

另外用本次实现对当前平台仓库执行了只读 discovery，基线为 `897c8bb7c752e24e172ef74b3846f2d2f16c2501`。导入 30 条历史变更记录与 6 条白名单平台记录，历史生成 30 条待策展草稿，平台文档未生成额外规则，输入错误为 0，批次到达 `awaiting_review`。详情见 [发现批次记录](C:/Users/irtos/AppData/Local/Temp/hmopt-evolution-self-discovery-20260910/batch-result.json)。

该发现实跑没有激活规则、确认候选或修改业务源码；扫描返回 `not_scanned_reason: no_active_patterns` 与 `coverage_complete: false`。因此它验证了真实仓库和已有平台记录的导入路径，不能解释为已经扫完整个目标内核或未发现优化机会。两份实跑状态均位于独立临时目录；需要长期保留时应连同 SQLite 数据库一起归档。

配置样例通过类型模型校验，OpenCode JSONC 示例去除注释后可解析，两份 V2 文档的本地链接已检查。Evolution 模块及测试通过 Ruff 核心规则、导入排序和格式检查；原有修改入口通过核心静态检查，Git diff 无空白错误。

最终全量本地回归：

```text
python -m pytest -q tests -k "not test_exec_command_not_found"
529 passed, 1 skipped, 1 deselected, 1 warning in 299.31s
```

其中 409 项为 Evolution 相关通过项，120 项为原有通过项。1 个符号链接测试因本地 Windows 创建权限限制跳过；1 个旧 fastboot 探测测试按上述范围排除。唯一 warning 来自故意将非法字典注入 Pydantic 模型的负例；测试确认重新校验会拒绝它。该结果覆盖本地协议和适配，不包含真实模型调用、目标内核构建、真机收益或团队发布验收。
