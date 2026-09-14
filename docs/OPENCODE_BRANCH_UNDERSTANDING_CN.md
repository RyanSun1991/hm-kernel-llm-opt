# opencode 主力分支：代码理解与 Evolution 接入判断

审计基线：`origin/opencode` / `c50d26e`。这是本次重新设计的代码基线。
此前 `codex/evolution-v2` / `057ce17` 基于 `main`，保留作设计与实现参考。

## 1. 项目的当前形态

本项目同时承载三条互相补充的链路：

1. **OpenCode 工作台**：日常对话默认进入 assistant；研究、方案、实现、评审、验证按通用角色组织。用户决定交接，任务工作区保存现场。
2. **显式自动化 recipe**：用户运行 `/optimize_*` 后，coordinator 通过 `task()` 调度通用角色，并执行方案、代码、真机测试阶段门。
3. **Python 平台与 MCP 服务**：索引、运行数据分析、Git、构建、测试、烧写、团队记忆通过服务提供；既有 LangGraph 优化循环仍保留。

Evolution 应作为第三条链路提供的候选与证据服务，并通过第二条链路执行；普通工作台对话不应被它自动接管。

## 2. 代码地图

| 层 | 主要代码 | 实际职责 |
|---|---|---|
| 公共入口 | `src/hmopt/cli.py`、`__main__.py` | run/optimize/index/query、pipeline、resolve/retrieval-eval、sediment/promote-skill |
| 配置 | `core/config.py`、`configs/*.yaml` | includes 合并、环境变量、索引后端、模型、存储与工作负载 |
| 通用角色 | `.opencode/agents/{assistant,researcher,architect,implementer,reviewer,validator,coordinator}.md` | 角色责任、工具上限、允许编辑的产物目录 |
| 技能与组合 | `.opencode/skills/_registry.yaml`、`skills/{infra,role,scenario}`、`agents/profiles` | 按需技能发现、领域方法、重复组合 |
| 工作区 | `.opencode/templates/workspace`、`scripts/new_workspace.sh` | task、capsule、decisions 与分角色工件 |
| 既有 pipeline | `opencode/pipeline.py`、`configs/pipeline_profiles.yaml` | 初始化/恢复 session、生成当前任务、校验 profile 引用 |
| 代码索引 | `indexing/models.py`、`backends`、`llamaindex_pipeline.py` | 统一 CodeIndex、clangd/SCIP、向量与图检索 |
| 运行数据 | `analysis/runtime`、`indexing/runtime_ingestion.py` | hiperf/flamegraph 等解析、热点与源码关联 |
| Agent 循环 | `agents`、`orchestration` | Conductor/Coder/Reviewer/Verifier/Profiler 与 LangGraph |
| 设备服务 | `api/auto_test_mcp_service.py`、`api/flash_mcp_service.py`、`tools/windows_device_relay` | IC、lmbench、Windows 中继、烧写与原始结果 |
| Team Memory | `sediment/journal.py`、`sediment/opencode_reader.py`、`memory/local_curator.py` | 个人记录、会话读取、经验提取与本地策展 |
| Skill Hub | `skillhub`、`api/skillhub_mcp_service.py`、`hm-skill-hub` | 检索、schema、staging、去重、冲突处理、评测、晋升与发布 |
| 质量契约 | `tests/test_opencode_golden_commands.py`、`scripts/lint_skill_registry.py`、GitHub workflows | 原有命令/profile 不漂移、技能引用正确、核心与 Hub 回归 |

## 3. 工作台与自动流程的边界

根 `AGENTS.md` 和 `.opencode/CLAUDE.md` 是薄契约；详细方法在技能中。
创建新角色的依据是责任/权限发生变化，优化方向变化应新增技能或组合。
因此 Evolution 继续使用 architect、implementer、reviewer、validator 和 coordinator。

工作台事实位于 `.opencode/local/workspaces/<slug>/`，capsule 是交接载体。
既有 pipeline 的 `.opencode/state/current_task.json` 是另一套运行实例状态。
Evolution 需要多候选并存，不能复用这个单例；每个 dispatch 建立独立 workspace。

角色权限是现有 frontmatter 中的 pattern-scoped maps，技能不能扩大权限。
方案作者和独立评审者必须是不同职责；同一个 architect 产出两份文件不构成独立评审。
服务端的 actor 是可信本地操作者声明，不是身份认证或强隔离的执行证明。

## 4. 索引能力已经存在

`CodeIndex` 统一包含 chunks、relations、file_summaries、relation_summaries、diagnostics。
两个后端均有实现与测试；部分文件注释中的 “SCIP planned” 已落后于代码。
SCIP 额外提供 callsite、读写 role bits、occurrence_count 等信号。
MCP 已能查符号、调用链、热点上下文和片段；Evolution 的语义复核应复用这些接口。

当前 Evolution 初筛是冻结 Git blob 上的字面匹配和热点权重，不等于 AST 证明或根因判定。
候选分数是排序启发式；语义索引辅助的 researcher/architect 复核、责任人判断、独立评审仍不可省略。
没有配置索引/真实热点时，必须明确缺少这部分证据。

## 5. lmbench 与 IC 的真实能力

lmbench 已有异步启动/status 与 Windows detached runner。结果包括 `total_result*.xlsx`，
其中 `result` sheet 的 `value0..N` 是原始数值，另有均值/波动与 digest。
现有 `vs_previous` 按文件时间选择参考结果，适合快速观察趋势；它没有证明“参考结果就是本方案 stock 镜像”。

正式量化门需要显式 stock/feature 配对、镜像/源码版本/设备/工作负载绑定、完整原始值、
预先冻结的主指标与 guardrails。一次 suite 内重复值先聚合为一次测量，不能扩充为多个独立刷机配对。
IC 原有 compare 产物保留了 case/round/step 的配对值，可复用严格 IC manifest 适配器。

`status=done` 说明测试进程完成，不等同功能通过、性能通过、代码可合入或技能已发布。

## 6. Team Memory 与 Hub 已形成原生闭环

原生链路：`write_entry` → journal → `journal_to_candidates` / sediment → schema-valid staging
→ central curator → 去重/冲突/评测/晋升 → release 与 lock pin。

六类记录覆盖已验证结论、专家裁决、结构事实、踩坑根因、方法与纠正。
没有通过的尝试不能伪装成 validated 经验；journal 自带的 outcome 是作者声明，
Evolution 导出前还要核对状态、评审、报告与内容哈希。

旧 Evolution SQLite 中 `kind=skill` 的 journal/staging/hub tier 是执行证据的内部成熟度，
不等于主线的个人记忆、正式知识编号或已安装技能。新接入保留审计记录，但明确标为
`execution_evidence` / `not_published`，并用薄桥接写入原生 journal 与 Hub staging。

## 7. 与旧分支方案的取舍

| 旧内容 | 处理 | 原因 |
|---|---|---|
| SQLite/WAL、版本 CAS、幂等 request、内容寻址证据 | 复用 | 与主线实例状态互补，适合候选级审计 |
| 历史挖掘、pattern 草案、所有者确认、冻结验证策略 | 复用并接主线来源 | 平台通用能力，不依赖旧 manager |
| os-opt-manager / kernel-* 调度 | 重写 | 主线已迁到 coordinator + 七角色 |
| 独立 task.json 指向任意服务目录 | 重写 | 角色只能写被允许的 workspace 工件，远程路径需共享可见 |
| 独立 JSON bundle 当成 Skill Hub | 替换主使用路径 | 主线已有 schema、curator、eval、release |
| 仅 IC 的转换入口 | 扩展 lmbench raw adapter | 主线已有 lmbench，不能丢弃其原始证据 |
| 覆盖 cli.py、pyproject 或索引模块 | 避免整文件覆盖 | 会丢失新命令、optional indexing extra、protobuf 等依赖 |
| 自动 pattern 自我晋升 | 保留人工策展门 | 负反馈可更新排序，真实共享技能还需原生评测与治理 |

## 8. 部署与验证注意点

核心安装 `pip install -e '.[dev]'`；重索引依赖通过 indexing extra 安装。
`run_all_mcp_servers.sh` 启动 Index、Seq、AutoTest、Flash、SkillHub；Build/Git 仍需对应服务。
Evolution 的 19 个工具已纳入 Index 所在的主 MCP：`api/evolution_mcp_service.py` 注册，
`api/mcp_service.py` 统一组合，stdio/HTTP 共用；已有 OpenCode 主 MCP 连接无需增加新服务。
通过 `HMOPT_EVOLUTION_*` 配置实例、工作区、工件和发现 profile，具体值见 Evolution 操作指南。
已有服务配置与 OpenCode provider 配置是不同对象，新示例不覆盖仓库当前 provider 设置。

本机离线验证可以覆盖状态机、真实临时 Git 提交、workspace 接入、原始 xlsx 解析和 native staging。
生产内核的索引完整性、OpenCode 模型实际执行、真机收益、并发设备租约及原生 Hub 发布，需要相应环境与证据。
测试夹具的 hardware 字段不构成真机实验记录。
