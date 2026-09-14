# Evolution on opencode：平台方案与实施计划

> 2026-09-11 更新：历史挖掘改为代码变更分析协议，由现有 OpenCode 模型读取前后代码并提交结构化结果。详见[历史挖掘设计与执行方式评估](EVOLUTION_HISTORY_MINING_DESIGN_CN.md)。已补齐 Skill 模式归并、候选语义复核、审批档案和显式串行执行批次，见 [Skill 工作流实施](EVOLUTION_SKILL_WORKFLOW_IMPLEMENTATION_CN.md)。后台并发挖掘 Worker 与生产质量评测另行建设。

基线：`origin/opencode@c50d26e`。本方案取代 main 基线上的接入假设；业务可落在内存、性能、可靠性或其他工程场景。
状态：原生适配已实现并完成离线集成验证；生产试点按第 9 节实施。验证记录见同目录操作指南。

## 1. 目标和约束

把一次优化扩展为可重复执行、证据可追溯的循环：

```mermaid
flowchart LR
  A[历史 Git / 工作区 / Team Memory / Hub] --> B[来源快照与 pattern 草案]
  B --> C[策展激活与有界全库初筛]
  C --> D[语义复核与责任人确认]
  D --> E[coordinator 显式 recipe]
  E --> F[方案 / 实现 / 独立评审]
  F --> G[lmbench 或 IC A/B / correctness 门]
  G --> H[执行证据与反馈统计]
  H --> I[原生 journal / Hub staging]
  I --> J[既有策展 / 评测 / 发布]
  J --> A
  H --> B
```

默认 assistant 工作台不自动开始此流程。责任人未确认的候选不能进入实现；通过验证不自动提交、合入或发布。
这次实现的是本地可信操作者环境中的服务和原生接入契约；不把 actor 字符串、SHA256 或测试返回状态称为身份/设备认证。

## 2. 分层设计

| 层 | 新增/复用 | 权威信息 |
|---|---|---|
| Evidence/Discovery | Evolution store、mining、sources、discovery | 来源版本、范围覆盖、pattern 版本、候选证据 |
| Workflow/Gates | EvolutionService | 候选版本、确认、已评审方案、Git 实现、代码评审、验证结果 |
| API/Transport | `api/evolution_mcp_service.py` 注册工具，`api/mcp_service.py` 统一挂载 | 与内核索引共用 stdio/HTTP 主 MCP；配置由服务端冻结 |
| Workbench execution | 既有角色 + 新 execution skill、dispatch workspace | capsule/工件是运行投影；状态晋级仍由 service 决定 |
| Evaluation | 既有设备采集 + IC/lmbench/correctness adapter | 原始输入、冻结 policy、规范报告与 verdict |
| Learning | 反馈统计 + native_memory bridge | 内部经验保留执行证据；共享真相归原生 Team Memory/Hub |

## 3. 来源与漏斗

Git 以解析后的 commit object 和第一父链分页读取。工作区只读允许的经验/方案/评审/验证工件；
不读取整个任意目录或把对话中的指令当作执行指令。
原生 journal 来源显式限定 contributor/project，Hub 只读 knowledge 范围。
每页有文件/字节预算，快照和 cursor 绑定；恢复失败、快照变化、无法推进时给出 attention/partition 状态。

pattern 表达问题、诊断路径、修复思路、字面谓词、适用条件、风险与来源。
历史代码先由工作台模型逐变更理解，Skill 再进行跨源归并和逐候选适用性复核；规则和引用通过服务校验后仍须策展/专家决策，不宣称已证明跨场景语义等价。
策展后显式激活；退役版本不继续筛选。

初筛顺序：路径范围 → 字面匹配 → 当前版本热点求交 → 责任归属 → 反馈权重与 Top-N。
源码使用 Git blob，工作区脏文件不会污染候选内容。
缺少热点、存在未消解历史假设或覆盖被截断时，应走工作台深入分析。
候选 score 不是“正确概率”，acceptance rate 也不是 precision。

超出本地预算的仓库/词典需要分区；当前实现会显式报告截断，不把前 N 个文件称为全库完成。
分布式索引扫描和跨库调度属于后续部署扩展，接口保持来源与覆盖可追溯。

## 4. 状态与权利

```text
discovered → confirmed → plan_approved → implemented → code_approved → validated
     │            │              │              │            │
     └────────────┴──────────────┴──────────────┴──────→ rejected
```

- confirm/reject：已绑定责任人通过 operator CLI/确认单操作；MCP 不提供 owner bypass。
- approve_plan：plan 绑定 candidate、baseline、精确 allowed_paths 和 policy；不同作者的 review 绑定 canonical digest。
- record_implementation：真实 Git 后代提交，变更路径属于批准范围；不能提交相同基线或评审人自己实现。
- approve_code：独立评审绑定 implementation digest；以后代码变化必须重新提交与评审。
- validate：报告、候选、实现、指标、环境一致；仅 inconclusive 可由责任人显式 retry_validation。
  fail 需分析原因并另走经评审的实验，不能直接重试到 pass。

CAS version 防止陈旧决策；request_id 只允许同一请求重放。修改参数必须用新请求；同 ID 不同输入返回冲突。
设备排队、源码隔离、人员账号由执行平台管理，不由一个 staged 文件授予权限。

## 5. 原生工作台接入

### 5.1 统一 MCP 边界

MCP 协议与注册逻辑归属 `src/hmopt/api/`。`evolution/` 保留领域服务、发现、状态机和验证逻辑，
不承担另一套默认传输入口。代码分工如下：

| 模块 | 职责 |
|---|---|
| `api/evolution_mcp_service.py` | 冻结 Evolution 服务配置，注册 38 个工具，按需初始化领域服务 |
| `api/mcp_service.py` | 把 5 个既有索引工具和 38 个 Evolution 工具注册到同一个主 MCP |
| `api/mcp_registry.py` | 对 Evolution 工具的顶层参数执行严格校验 |
| `api/mcp_stdio.py`、`api/mcp_server.py` | 复用同一注册表的 stdio 与 HTTP 入口 |
| `evolution/mcp.py` | 旧独立入口的兼容薄桥，不另维护工具定义 |

OpenCode 沿用已有主 MCP 连接、服务名与配置键；本地启动 `python -m hmopt.api.mcp_stdio`，
远程沿用 `hmopt.api.mcp_server:app` 的 `/mcp`。Evolution 默认注册，HTTP 沿用主服务鉴权。
不增加新的 Evolution 端口、默认 OpenCode 连接或 provider 配置。
首选通过 `HMOPT_EVOLUTION_CONFIG` 加载单文件配置，CLI 用 `--config` 指向同一文件。
`setup --repo ... --owner ...` 生成配置、内联 pilot profile 与供合并的 OpenCode fragment；
`doctor` 检查配置、Git、工具和组件就绪，不运行模型、设备或候选流程。
setup 的 owner 是操作者明确指定的试点责任人，不自动推断专家身份，也不发送通知。
五环境变量的进阶/兼容方式继续支持 `HMOPT_EVOLUTION_ROOT`、`HMOPT_EVOLUTION_GIT_BIN`、
`HMOPT_EVOLUTION_WORKSPACE_ROOT`、`HMOPT_EVOLUTION_ARTIFACTS_ROOT` 和
`HMOPT_EVOLUTION_DISCOVERY_PROFILES` 配置实例与路径。
配置和 discovery profiles 在启动时冻结；仅启动、列工具或查看 profiles 不创建数据库，
首次需要 store 的调用才初始化实例。CLI 和 MCP 必须使用同一个 state root。
配置就绪、候选通过证据门和生产实测收益是三个层次，doctor 通过不能证明后两者。
最小操作路径见[三步快速开始](EVOLUTION_QUICKSTART_CN.md)。

### 5.2 工作台执行契约

`/evolve-discover <profile> [batch-id]` 是有界发现入口；服务器注册的 profile 固定路径、责任规则和预算。
统一主 MCP 中的 38 个 Evolution 工具覆盖批次发现、独立 mine/sources/distill/scan、记录查询、分发、证据提交与验证，以及多仓总任务、断点续扫和实验准备。
未知工具参数显式拒绝，Agent 不能传扫描路径或 running 恢复开关。入口、配置和工具表见操作指南第 2 节。

`/evolve-candidate <candidate-id|absolute task.json> [scope]` 是候选执行入口。
scope 可为 full、stage、status、research、plan、implement、review、validate；后者只执行当前状态允许的步骤。
coordinator 获取最新 handoff，与 immutable dispatch 比较版本和摘要。
先前的 `/optimize_*` recipe 和 golden 契约保持其既有语义。

| service 下一阶段 | task 步骤 | 输出责任 |
|---|---|---|
| architect | researcher → architect → clean-context reviewer | 研究、方案、独立方案评审分别产出 |
| implementer | implementer | 精确范围代码修改、真实提交、实现记录 |
| reviewer | reviewer | 针对实现 digest 的独立代码评审 |
| validator | validator | 功能证据、显式 raw manifest、规范报告 |

immutable 包保存在 Evolution store 的 dispatch 目录；角色产物进入
`<project>/.opencode/local/workspaces/evolution-<dispatch>/`，使用角色已有目录上限。
capsule 记录候选、dispatch、阶段和下一步；不得覆盖 singleton current_task。
重试可修复未完成的生成过程，不覆盖角色已经更新的 mutable capsule/task 工件。
仅导出包但未绑定 workspace 时不能开始执行；远程 MCP 必须配置 agent 与服务均可访问的共享路径。

每个阶段完成后通过服务提交产物，再取得下一阶段的新 dispatch。不能用文本“approved”代替真实 gate。
模型调用由已安装的 OpenCode runtime 执行，Python dispatch 本身不伪装成已经运行了 Agent。

基础确认可通过 CLI/确认单执行。可选生产审批桥已提供责任人目录、SMTP TLS/webhook 投递、
认证网关回调及批准 ID 归档；企业登录与专家页面由接入网关负责。确认后的 continuation
仍需显式选择执行范围，不自动唤醒会话。当前配置与协议见
[审批桥实现](EVOLUTION_APPROVAL_BRIDGE_DESIGN_CN.md)和[生产使用指南](EVOLUTION_PRODUCTION_RUNBOOK_CN.md)。

## 6. 量化与正确性验收

性能 policy 预先冻结一个主指标、可选 guardrails、方向、单位、最小收益、允许回归、最低配对数与环境。
按指标方向统一计算相对改进及配对统计；功能失败、显著 guardrail 回归、证据不足分别产生 fail/inconclusive。
优化 memory-bound 路径时主指标可以是延迟/带宽/页面行为，不被 IC 默认目标覆盖。

lmbench 适配器要求显式 suite pairs：每个 baseline/candidate 文件、SHA256、run_token 与测量上下文。
原始 xlsx 的 system/tool/metric/command/unit 完整匹配，内部 valueN 聚合为一次 suite 样本。
同一个文件或运行 token 不能重用为独立配对；不从 digest、mtime“前一次”或 PASS 文本制造报告。
输入哈希、manifest、规范报告在同一转换记录中关联；validation 冻结这些来源引用。
方案中的 `measurement_profile_sha256` 提前冻结原生指标身份映射。validate、quality 和 native export
都核验对应来源，直接提交普通 ABReport 不能绕过 profile。

correctness 使用独立 policy：基线复现失败、候选必需检查通过、环境与版本一致；不制造性能百分比。
simulation 可用于协议演示，不能进入有效收益统计和原生 validated 导出。

## 7. 原生沉淀与下一轮反馈

每轮记录成功、失败、专家否决和反例；已有指纹抑制重复推送，人工 overlay 控制排序和 probation 路由。
统计同时展示责任人选择率、真实通过/失败/不确定、模拟与无效报告；未知 precision 保持 unknown。

内部 `skill` 是执行证据成熟度投影。原生导出必须验证源报告/评审完整、内容可共享、curator 独立且明确选择当前条目。
桥接只把本条已审文本写入 native journal，调用原生转换/schema 校验后写 Hub staging，不重扫整个个人 journal。
结果保存 journal ID、staging 路径、schema/哈希及幂等请求；最终共享知识 ID、技能评测、发布版本仍由原生工具生成。

## 8. 本次实施工作包

| 工作包 | 交付内容 | 验收 |
|---|---|---|
| A 基线审计 | 主线理解、旧方案取舍、API/schema 盘点 | 指向实际代码，保留主线命令与依赖 |
| B 核心与统一 API | Evolution 领域模块 + CLI 子命令 + `api/` 主 MCP 工具注册 | 旧核心门禁、主 MCP 工具共存、统一 stdio/HTTP、旧入口兼容与惰性初始化 |
| C Workbench | 通用角色 recipe、registry、workspace materialization | 原有 golden 不漂移、新 recipe 与过期 dispatch 测试 |
| D 原生来源/沉淀 | workbench 白名单、journal/Hub 分页输入、native staging | 范围隔离、重试、伪造/模拟拒绝、原生 schema |
| E lmbench | 原始 suite 配对转换与来源证据注册 | 正/负向、guardrail、证据关联、IC 兼容 |
| F 操作闭环 | 配置示例、运行手册、离线端到端验证 | 可复制命令、实际日志、清晰列出环境依赖 |

## 9. 生产落地顺序

1. 在目标内核 checkout、OpenCode runtime、索引 MCP 和共享 artifacts 路径上配置；先做只读 discovery。
2. 选择一个责任明确、已有可复现测试的候选，完成真实独立方案评审与隔离实现；记录耗时和人工介入点。
3. 配置真机设备/镜像与工作负载，采集至少冻结策略要求的独立 A/B suite pairs，保留所有原始工件。
4. 达到门槛后通过原生 Hub staging/curator/CI；选择下一候选检验经验召回是否真实改善任务。
5. 累积数据后再扩大历史/仓库分区、并发执行与设备资源池；以每个有效候选成本、真实收益、回归率和复用效果评估。

这些生产步骤需要真实环境，不能由离线 fixture 的 pass 替代。

| 批次 | 责任角色 | 进入条件 | 退出证据 |
|---|---|---|---|
| P0 配置验收 | 平台维护者 | OpenCode、MCP、路径/账号可用 | 同一状态实例、可读索引、可恢复workspace |
| P1 单候选灰度 | 模块责任人 + 五个执行角色 | 一个责任明确且有测试的候选 | 实际方案/代码双评审、Git提交与真机完整报告 |
| P2 沉淀复用 | 独立curator + Hub维护者 | P1证据完整且文本可共享 | 原生staging通过CI、策展结论、下一任务召回记录 |
| P3 批量扩展 | 平台/测试设施维护者 | 重复试点证明收益与可恢复性 | 分区coverage、设备并发/租约、有效候选成本与回归率报表 |

逐批放量以退出证据为依据，尚未提供的生产环境不填写虚构完成日期。

## 10. 隔离和信任边界

角色沿用已有 permission ceiling。多个角色共享 `.opencode/local/**` 的可写上限，分角色工件子目录
是协作协议，不是文件系统级安全隔离；强隔离和身份认证需要部署层执行器支持。
提交门校验作者区分、状态版本、源码提交和内容摘要，不能证明两个 actor 字符串对应真实独立的人。
上述边界应在单团队可信操作者部署中理解，不能直接推广为多租户认证能力。

生产后台挖掘、专家通知与认证回调、质量评测的配置及使用见 [生产增量指南](EVOLUTION_PRODUCTION_RUNBOOK_CN.md)。
