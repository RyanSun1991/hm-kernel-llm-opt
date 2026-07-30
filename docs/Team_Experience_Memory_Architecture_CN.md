# HM OpenCode 团队经验记忆与 Skill Hub 融合架构设计

| 项 | 值 |
|---|---|
| 状态 | Proposed final design（评审修订版） |
| 版本 | v2.0 |
| 日期 | 2026-07-30 |
| 基线 | RyanSun1991/hm-kernel-llm-opt · opencode 分支 · 210c73c |
| 取代 | v1.0 英文稿《HM OpenCode Team Experience Memory and Skill Hub》；`docs/Hub_Lite_Universal_Access_Design_CN.md`（其内容已归并至本文） |
| 读者 | HMOPT 维护者、OpenCode 集成工程师、Skill Hub 维护者、平台/安全工程师 |
| 语言策略 | 散文中文；路径/字段/代码/API 一律英文 |

**v2.0 相对 v1.0 的修订**（对应评审发现的六项问题）：

1. 新增「规模分档」（§4）：S/M/L 三档拓扑，默认 S 档，升档需触发条件——解决"平台级设计与团队现实错配"；
2. 新增「复用映射」（§6）：关系分类、检索、技能归纳、沉淀四处强制复用已验证资产（七类判定器 48 例基准、混合检索 26 查询基准、GEPA/评测门管线、sediment 链），并给出 v1.0 的 11 项关系清单与七类判定的归并表——解决"重写已验证资产"风险；
3. 成熟度语义修正（§7.2）：L0–L3 维持主设计语义、仅适用于知识与技能；event/episode 是**层**不是成熟度——解决 L0 语义漂移；
4. 新增「Schema 与版本策略」（§12）：新 schema 家族随 hub 1.0.0 major 发布，量化对 `release.py` 哈希规则的冲击——解决"schema 冲击未量化"；
5. 过度设计项降档（§11.5）：本地静态加密、法务 hold、tenant 层级标记为策略可选/预留——解决"局部过度设计"；
6. 新增人力估算（§14）：每期人周数与团队假设——解决"无成本锚点"。

前置整改项（凭据轮换、端口注册、sediment 覆盖/`subsystems[0]`/human_decisions 三处既有缺陷修复）**另行跟踪，不在本文展开**；本文假定其在 Phase 1 开始前完成。

────────

## 1. 问题与目标

### 1.1 问题

工程开放给全团队后，用户形态分化：跑完整优化 pipeline 的是少数；多数人做研究、查一个函数、或者只是把 OpenCode 当通用编码助手。现状是：

- 只有 pipeline 和特定 agent 会写出可沉淀的结构化产物；普通会话的调试发现、工具用法、失败教训、用户纠正全部留在单次对话里；
- 捕获依赖模型自觉（human-interaction-memory 技能是"社会约定"不是"运行时保证"），错误、中断、压缩、换 agent 都会丢事件；
- 默认路由过重，普通提问也继承优化管线人格；
- 本地记忆格式分裂：聚合 markdown 可被 sediment 读取但不可被 Resolver 叠加检索；
- OpenCode 会话、pipeline 任务、HMOPT run 之间无统一关联模型。

### 1.2 双重目标

1. 让记忆积累对普通用户**接近无感**；
2. 让原始的、敏感的、嘈杂的、未验证的对话**绝不直接成为**可信团队知识。

单靠 prompt/skill 做不到第一条（无法保证捕获完备性），单靠自动捕获做不到第二条（遥测不是知识）。因此必须分层。

### 1.3 核心运行原则

> **广泛而本地地捕获；增量地提炼；选择性地共享；保守地晋升；带证据地检索。**

> **插件负责观察，本地守护进程负责持久，网关负责身份与策略，MCP 提供语义访问，蒸馏器负责提议，Skill Hub 决定团队可以信任什么。**

### 1.4 非目标（首个发布不做）

不上传全量原始转录；不自动合并候选入 Hub；不允许会话 agent 未经授权发布/删除团队知识；不用向量库取代 Git/Markdown Hub；不把 LLM 摘要当独立证据；不因单次成功改写共享技能；不承诺分布式 exactly-once；不因文件被打开过就存全文；不从出现频率推断真伪。

### 1.5 端到端成功判据

```text
用户 A 解决一个真实任务
  -> 事件本地捕获并对账
  -> 产出脱敏的、证据挂钩的候选
  -> 审核发布
  -> 用户 B 后续自动检索到该记录
  -> B 更快成功或避开已知失败
  -> 结果反馈挂回该记录的精确版本
```

────────

## 2. 术语与分层

| 术语 | 含义 |
|---|---|
| Event | 不可变观测：一条消息、一次工具调用/结果、一次文件编辑、一次验证结果、一次人工裁决 |
| Session | OpenCode 原生会话容器；一个会话可含多个目标 |
| Episode | 目标导向的会话切段：问题、约束、尝试、决策、证据、结论、未决问题；一个会话可产出多个 episode |
| Candidate | 结构化、证据挂钩的长期记忆提案；尚非可信团队知识 |
| Knowledge record | 受治理的事实/规则/启发式/反模式/验证陷阱/环境约束/可复用方法 |
| Skill | 改变 agent 行为的过程性指令包；比知识需要更强的评测与不回退控制 |
| Skill Hub | 现有 Git/Markdown 治理层：schema、七类判定、策展、评测、发布、钉版 |
| Memory Control Plane | 检索/取全文/反馈/检查点/状态/策略的逻辑服务；MCP 是其一种接口 |
| Experience data plane | 高频事件捕获与传输通道，与模型可见的 MCP 工具面刻意分离 |

**分层 vs 成熟度（本版关键澄清）**：Event/Episode/Candidate/Curated/Skill 是**生命周期层**（东西是什么）；L0–L3 是**成熟度**（多可信、可复用多广），维持主设计语义——L0 草稿、L1 候选、L2 稳定、L3 核心——且**只适用于 knowledge 与 skill**。事件与 episode 没有 L 级。

────────

## 3. 架构总览

### 3.1 三个平面、五个运行时职责

| 平面 | 职责 | 组件 |
|---|---|---|
| 经验数据面 | 捕获与传输事件 | OpenCode 捕获插件、ingest API |
| 记忆控制面 | 检索、检查点、策略、反馈 | Memory MCP、Gateway |
| 知识治理面 | 策展与发布稳定知识/技能 | 现有 Skill Hub |

| 运行时职责 | 组件 |
|---|---|
| 确定性观察 | `hm-memory-opencode` 插件 |
| 持久、对账、本地策略 | `hm-memoryd`（S 档可进程内实现，接口保持守护进程边界） |
| 高频传输 | 认证 ingest API |
| LLM 可见语义操作 | Memory MCP facade |
| 策展真相源 | 现有 Skill Hub（Git/Markdown） |

### 3.2 逻辑架构

```text
用户 → OpenCode ←─ 全局规则 + 薄技能（只写"何时检索/引用/反馈"）
          │
          ▼
   hm-memory-opencode 插件 ──追加──▶ 本地 SQLite WAL + 内容寻址 blob
          │（回调只做一次本地事务，p95 < 10ms，绝不联网/跑 LLM）
          ▼
      hm-memoryd：对账（以 OpenCode SDK 会话为准）、配对工具调用、
      深度脱敏、留存、episode 构建、离线重试、本地上下文缓存
          │                                    ▲
          ▼ ingest API（批量、幂等、游标）        │ memory_context / memory_get（MCP）
      Memory Gateway：身份、ACL、配额、审计、workspace 映射
          │
          ▼
      事件/episode/candidate 存储 → 蒸馏器 → 关系判定（七类内核）→ 策展
          │
          ▼
      人工评审 + PR → Skill Hub 发布（semver + 不可变 commit）→ 检索索引重建
```

### 3.3 信任边界

```text
用户机器：OpenCode、插件、本地加密存储（策略项）、hm-memoryd、原始内容默认只在此
团队服务：gateway、ingest/MCP 端点、ACL 分区的存储与索引
治理边界：策展器、评审人、Git PR、Hub 发布
```

每跨一层边界，策略更强：本地捕获策略 → 原始上传/脱敏策略 → 显著性+证据门 → schema/关系/评审/晋升门 → 技能评测门。

### 3.4 降级语义

| 故障 | 行为 |
|---|---|
| MCP 不可达 | 任务继续；允许时用本地缓存；记录一次检索失败 |
| ingest 不可达 | 本地排队重试；绝不阻塞 OpenCode |
| Skill Hub 不可达 | 只检索个人/项目记忆；标注 Hub 不可用 |
| 蒸馏器不可达 | 事件与 episode 留存待处理 |
| 身份过期 | 暂停远端同步；本地捕获按当前策略继续 |
| 本地磁盘满 | 显式警告；停止确认捕获；**绝不静默丢弃** |
| 插件被禁用 | OpenCode 照常；`memory_status` 报告捕获不可用 |

────────

## 4. 规模分档（v2.0 新增）

**纪律：默认 S 档。升档需要命中触发条件并过一次架构评审；禁止未命中触发条件就搭 L 档组件。**

| 档 | 适用 | 拓扑 | 明确不建 |
|---|---|---|---|
| **S（单机/试点，默认）** | ≤10 人、单团队、内网 | 插件 + 本地 SQLite/进程内 memoryd + 单实例 memory 服务（复用现有 FastMCP 工艺）+ 现有 Git Hub；文件系统存 blob；进程内 worker | 网关集群、Postgres、对象存储、消息队列、租户体系 |
| **M（团队试点）** | 10–50 人、1–3 团队 | + OIDC/内部 SSO 网关、中心元数据库（Postgres）、可选对象存储、worker 队列 | 多租户、跨区部署 |
| **L（多团队生产）** | >50 人或跨部门 | + NATS/Kafka、分区索引、企业密管、append-only 审计库 | — |

升档触发条件（任一命中才允许）：并发用户数超过档位上限；单机存储/检索延迟不达 §15 指标；出现第二个团队的隔离需求；安全评审要求集中审计。

v1.0 §27 的「首个垂直薄片」保留为**强制首切片**（见 §14 Phase 1），它完全落在 S 档内。

────────

## 5. 组件设计

### 5.1 team-assistant 默认路由

新增轻量主 agent `team-assistant` 作为默认路由：普通编码/答疑助手；实质性任务先检索记忆；引用影响结论的记录 id；验证结果后提交反馈；不加载 research/plan/implement 硬门禁。专用路由保持显式选择：`@kernel-research`、`@kernel-plan`、`@hm-opt-manager`。

### 5.2 全局规则与薄技能

全局规则（managed block 写入 `~/.config/opencode/AGENTS.md`，兼容 `~/.claude/CLAUDE.md`）只讲五件事：何时检索、如何用适用性与证据、如何引用 id、何时反馈、如何尊重隐私控制，并声明"捕获由插件自动完成，不要把对话手工抄进 Git"。详细逻辑在服务与技能里，不在每个人的 prompt 里。

**捕获两步走（归并 Hub-Lite 立场）**：插件是捕获的最终答案（ADR-001）；在插件落地前的过渡期，允许"契约 + 客户端钩子"作为临时捕获通道；插件可用后，契约**降级为纯引导层**（何时检索/反馈），不再承担捕获职责。

### 5.3 hm-memory-opencode 插件

- 观察事件：会话创建/更新/压缩/空闲/错误/关闭；用户与助手消息修订；工具执行前后；文件编辑与会话 diff；显式反馈与 /memory 命令；当前 agent/模型/仓库/分支/commit/worktree。
- 回调只做：构造小信封 → 一次本地事务 → 返回。目标本地追加 p95 < 10ms。回调内**禁止**网络、embedding、LLM、深度脱敏。
- 去重键：消息 `(opencode_session_id, message_id, part_id, source_revision)`；工具 `(opencode_session_id, tool_invocation_id, phase)`。本地生成 UUIDv7/ULID 作为 canonical event_id。
- 立即生效的本地控制：暂停捕获、隐私模式、"不要记录"区域。

### 5.4 hm-memoryd

职责：本地事件日志与上传 outbox；每会话有序序号与游标；**以 OpenCode SDK/API 的完整会话为准做对账**（增量钩子丢失/修订的事件补齐，重复事件靠唯一约束忽略，未配对工具调用标记或后配对）；深度脱敏与策略分类；内容寻址 blob；本地 episode 构建；留存与删除；退避重试；确认持久化；离线上下文缓存；状态暴露；调试导出。

存储：`$XDG_DATA_HOME/hmopt-memory/{memory.db, blobs/<sha256>, exports/, logs/}`，SQLite WAL。**静态加密为策略项**（§11.5）。

S 档允许进程内实现，但接口与数据模型必须保持守护进程边界，便于后续拆出。

### 5.5 Memory Gateway

MCP 与 ingest 的唯一策略执行点：认证（M 档起 OIDC/内部 SSO；S 档允许内网短期令牌）；绑定 org/team/user 与注册 workspace；检索前与写入前评估 ACL；配额限流；schema/适配器版本校验；workspace→许可根目录映射；审计；**拒绝任意服务端文件路径**。身份规则：服务端只信认证声明，模型提供的 user_id/team_id/路径最多作参考，不构成授权。

### 5.6 ingest API（非模型工具）

```text
POST /v1/events:batch     POST /v1/episodes:close   GET  /v1/sync/cursor
POST /v1/deletions        POST /v1/candidates:submit  POST /v1/blobs
```

批语义：workspace/session 标识、首末本地序号、不可变 event id、内容哈希、适配器与 schema 版本、幂等键、可选 blob 引用。响应：最高连续接受序号、拒绝原因、服务端游标、策略/版本告警。已确认批次重放无害。

### 5.7 Episode 构建器

按**用户目标**切段而非按会话。闭合条件：目标实质变化；显式检查点；达成验证结果；有效工作转空闲；会话压缩/关闭；pipeline 阶段完成；可见性变化前策略要求。产出：目标、约束、路径/符号/target/子系统、尝试与状态、决策与理由、客观证据、用户裁决、检索使用情况、结论与置信、可复用断言与反模式、未决问题、源事件区间与哈希、隐私状态。

### 5.8 蒸馏器（两段，复用 sediment 链）

- 确定性段：时间线、命令退出码、构建/测试/基准结果、工具调用配对、变更路径与哈希、显式用户接受/拒绝/纠正、落地/回退 commit、既有产物格式、脱敏结果、检索上下文与被引 id——**复用 `sediment/extractors.py` 工艺，新增 event/episode reader**。
- 语义段：schema 约束下的 LLM 提议（问题摘要、方法、理由、适用与排除、可复用断言、未决不确定性、候选关系）——**复用 `sediment/salience.py` 工艺**。LLM 输出是提案，**永远不是证据**。

### 5.9 关系判定与策展（七类内核，见 §6.1）

候选评审期间可变，决策历史 append-only。策展器执行 schema 校验、脱敏与来源许可、证据充分性、适用性评审、关系分析、成熟度与可见性策略、人工评审路由、知识/技能治理分流。发布器：每记录一 Markdown 文件、开/更新 PR、跑 Hub CI 与评测、记录评审决定、semver + 不可变 commit 发布、重建索引、发出发布与失效事件。

### 5.10 检索索引（派生缓存）

索引永远是派生物，从个人/项目规范记录与已发布 Hub 记录重建。**基座为现有混合栈**（§6.2），排序特征做加法。

### 5.11 遗留适配器

聚合 markdown（targets/subsystems/global_lessons/idea/bad-plan/human-decision）→ 规范"一记录一文件"本地记录；既有 plans/reviews/bench/state 摄入；`skillhub_*` → 新服务映射；pipeline 任务与 OpenCode 会话关联；迁移期保留现有 Hub schema。

────────

## 6. 复用映射（v2.0 新增，强约束）

> **本节为实现约束：以下四处必须基于现有已验证资产扩展，禁止并行重写。**

### 6.1 关系分类：七类判定器是唯一分类内核

v1.0 列出的 11 项"关系"与现有七类判定的归并：

| v1.0 清单项 | 归并语义 |
|---|---|
| novel | 七类判定的默认分支「全新」 |
| duplicate | 关系「重复」 |
| additional evidence | **不是独立关系**：「重复」的处理结果（出处合并+确认数+1）；跨口径证据对应关系「口径差异」 |
| merge | **策展动作**，非关系 |
| supersedes | 「矛盾」「过时」的处理结果，非独立关系 |
| subsumes | 关系「泛化包含」 |
| contradicts | 关系「矛盾」 |
| temporal change | 关系「过时」 |
| conditional divergence | 关系「条件分歧」 |
| selector / scope drift | 关系「位置漂移」 |
| unrelated | 「全新」且不建关系边 |

即：11 项 = **7 个关系 + 2 个策展动作 + 2 个结果别名**。实现上：`local_curator.py` 的 `classify_relation/apply_relation` 为分类内核；策展动作由 `central_curate.py` 层执行；**48 例分类基准继续作为回归门**，为 chat 来源与"独立确认判别"（同会话重复/助手复述/同用户重复/重放再生不算独立确认）**向上加例，不改既有用例语义**。

### 6.2 检索：现有混合栈为基座

保留：scalar 预过滤（status/maturity/scope）→ BM25 + 向量余弦 → RRF 融合（k=60）→ 符号加分 → 晋升分 sigmoid 加权；**26 查询检索基准（must-recall@5 与消融）继续作为回归门**。新增排序特征做加法：证据强度、独立复用、新鲜度/有效期、矛盾/陈旧/有害反馈惩罚、项目/用户亲和（P2）。排序模型初期保持可检查、可调试的确定性形式。

### 6.3 技能归纳：对接现有引擎 B，不另起炉灶

v1.0 §13.7 的 skill induction 与现有技能进化管线是同一件事，映射为：

```text
重复过程证据（≥2 独立 episode/贡献者） → promotion_detector 提名
  → 有界技能补丁提案                      → skill_optimizer（SkillOpt 有界编辑 + GEPA 反思进化）
  → 代表集+留出集评测、不回退检查          → run_evals + eval_gate（严格变好且零退化）
  → 多候选保留                            → pareto 前沿
  → 自动合并信任                          → auto_merge_gate（≥3 次连续改进零回滚才解锁）
  → 评审批准、版本化发布                   → nightly 七步 + release/broadcast
```

### 6.4 沉淀：extractors/salience/validate 复用

新增两个 reader（experience_reader 处理 event/episode、human_decision_reader 处理人工裁决），接入现有 `sediment` 管线；schema 校验、候选打包、staging 纪律全部复用。

### 6.5 资产复用清单

| 新组件 | 必须复用的资产 | 扩展点 |
|---|---|---|
| 关系服务 | `local_curator` + 48 例基准 | 独立确认判别、chat 来源用例 |
| 检索索引 | `HybridRetriever` + 26 查询基准 | 反馈/新鲜度/证据排序特征 |
| 技能归纳 | SkillOpt/GEPA/Pareto/eval_gate/auto_merge_gate/nightly | 重复过程检测输入源扩到 episode |
| 蒸馏器 | `sediment` extractors/salience/validate | event/episode/human_decision reader |
| 发布 | `release.py`/`broadcast.py`/CI 五道门 | 新 schema 家族（§12） |
| MCP 服务 | `skillhub_mcp_service` 工艺（FastMCP、静默降级） | memory_* 工具面 |

────────

## 7. 数据与知识模型

### 7.1 治理四轴（独立，不互相推导）

```yaml
maturity: L0 | L1 | L2 | L3            # 多可信（仅 knowledge/skill）
visibility: personal | workspace | project | team | organization
sensitivity: public | internal | confidential | restricted | secret
sharing_state: private | project-candidate | team-candidate | curated-team
status: active | tentative | deprecated | superseded | contradicted | quarantined | tombstoned
```

### 7.2 生命周期层与成熟度（修订）

| 层 | 用途 | 默认位置 | 默认共享 | 成熟度 |
|---|---|---|---|---|
| Event | 完整观测 | 本地存储 | 私有 | **无 L 级** |
| Episode | 目标导向经验 | 本地/项目 | 私有 | **无 L 级** |
| Candidate | 结构化可复用断言 | 候选库 | 评审控制 | L1（schema 齐全+初步证据） |
| Curated knowledge | 稳定团队知识 | Skill Hub | 按 ACL | L2；跨团队复用后 L3 |
| Released skill | 评测过的过程行为 | Hub skill 发布 | 按 ACL | L2/L3 |

晋升判据沿用主设计：证据强度、schema 完整、脱敏与许可、独立贡献者/任务、矛盾解决、适用清晰、复用结果、评审批准。

### 7.3 事件信封 / Episode / 候选（要点）

事件信封不可变、紧凑，大内容按哈希外置；必含：schema/适配器版本、稳定 event id、来源 OpenCode id 与修订、本地序号、workspace 与 Git 上下文、事件类型与 actor、载荷摘要/哈希/引用、当时的 context pack、隐私/敏感/留存策略、父子/trace 关系、幂等键（完整示例见附录 A）。

Episode 必须区分：提出 / 尝试 / 用户接受 / 用户拒绝 / 构建通过 / 测试通过 / 基准改善 / 补丁落地 / 事后回退 / 结果未知——**防止模型乐观被当成事实晋升**。

候选类型（初始集）：事实/观察、决策与理由、启发式/规则、反模式/失败方案、验证陷阱、可复用方法/工具配方、环境约束、想法、未决问题、过程候选——发布时投影到现有 Hub schema（memory_item/global_lesson/bad_plan/idea）。

### 7.4 证据分级（LLM 置信度只是元数据）

1. 可重复的客观测试/构建/基准/生产结果；2. 落地或回退的实现结果；3. 显式人工接受/拒绝/纠正；4. 其他任务/贡献者的独立成功复用；5. 评审过的静态代码事实；6. 未独立验证的工具输出；7. 助手自评（最弱）。

### 7.5 血缘与删除图

`event -> episode -> candidate -> curated record -> skill evidence` 全程可追，支持：来源检查、纠正传播、删除/隔离、候选再生、被取代时的影响分析。

────────

## 8. 检索与上下文组装

- 命名空间顺序：会话工作记忆 → 个人 → workspace/项目 → 团队 Hub 知识 → 已发布技能；**ACL 过滤先于排序**（索引保持分区，不允许全局检索后再过滤）。
- 请求模型：query、intent（debug-build/research/plan/implement/review/general）、mode、workspace/revision、多 paths/symbols/targets/**多 subsystems**、namespaces、include、记录数与 token 预算。
- 候选过滤：ACL/可见性、敏感策略、状态、有效期、显式适用约束、仓库/版本兼容、环境条件；bad plan 与矛盾可作为**警示**保留返回。
- compact + exact fetch：`memory_context` 返回紧凑摘录，`memory_get` 取全文——限定 prompt 体积、留下使用遥测、降低误注入。
- 每条结果必含：id 与版本、类型、标题与摘录、适用与排除、失效条件、成熟度与置信、证据摘要、出处引用、内容哈希、**入选原因**、冲突/被取代状态。
- 冲突处理：预算允许时两条都返回、标注冲突、说明区分条件，不靠排名静默二选一。
- 预算：普通任务 1000–1800 token；研究 2000–3500；pipeline 交接按阶段（沿用现有 STAGE_BUDGET，新增 `chat` 档）；status < 300。
- 注入边界：检索知识以定界、带类型的**不可信数据**注入（`MEMORY RECORD - UNTRUSTED ... END MEMORY RECORD`）；只有已发布签名技能可改变过程行为；知识里的指令样文本忽略并记日志。

────────

## 9. MCP 工具与 API（命名拍板）

模型可见工具面保持小：

| 工具 | 用途 |
|---|---|
| `memory_context` | 排序后的紧凑上下文（含 context_pack_id、hub/index 版本、记录、技能、坏招警示、冲突、告警） |
| `memory_get` | 按 id/版本取全文、证据、出处、关系 |
| `memory_checkpoint` | 显式语义检查点：用户确认的事实、重要决策、验证结果、纠正、episode 闭合请求（**不是**事件传输通道） |
| `memory_feedback` | helpful/harmful/stale/contradicted/inapplicable/unused/accepted/rejected/corrected/test-pass/test-fail/benchmark-improved/benchmark-regressed |
| `memory_status` | 身份与 workspace、捕获状态、同步滞后、待传/候选数、Hub 版本与 commit、服务健康 |

兼容别名（迁移期保留 ≥2 个 minor 版本）：`skillhub_resolve → memory_context`；`skillhub_sediment → 遗留产物收割 + checkpoint`；`skillhub_status → memory_status`。

破坏性/隐私敏感操作走确定性命令而非模型自由决策：`/memory status|private|team-candidate|pause|resume|checkpoint|show-candidates|why <id>|forget-session`。

────────

## 10. 用户体验与配置

一条命令安装：`hm-memory install --team <url>` —— 认证、装插件与薄技能、managed block 更新规则文件（`<!-- hm-memory:begin/end managed -->`，不覆盖无关内容）、注册远端 MCP、初始化本地存储、询问默认留存与共享偏好、注册 workspace、自测、显示改动清单、提供 `uninstall/doctor`。

项目配置：全局只载 base 规则；`team-assistant` 为默认 agent；优化管线规则留在专用 agent/命令里；密钥一律 `{env:...}` 引用。

状态可见：`Memory: local capture on | team candidates allowed | synced | Hub 0.3.1`；捕获不持久、同步暂停、磁盘满、认证过期、隐私会话、删除待处理、Hub 过期/不可用时必须显式告警。

────────

## 11. 安全、隐私与治理

### 11.1 威胁模型（覆盖）

凭据/密钥捕获、私有代码泄露、跨团队检索、任意路径访问、投毒记忆、存储内容 prompt 注入、模型发起的破坏操作、未授权晋升、重放与重复、陈旧/矛盾建议、第三方许可内容、删除不传播、本地设备/令牌失陷、恶意贡献者刷确认数。

### 11.2 关键规则

- 认证：M 档起 OIDC/内部 SSO 短期令牌；共享静态令牌不允许用于生产。
- 授权：ACL 先于检索；命名空间 `personal/workspace/project/team/organization`；索引保持分区。
- 路径遏制：服务端只认注册 workspace 到许可根的映射，绝不接受模型提供的绝对路径作为授权。
- 原始内容策略：原始消息与工具载荷**默认仅本地**；脱敏 episode 元数据可选上行；候选评审控制；策展知识按 ACL。
- 脱敏：高风险字段在本地持久化前尽量先脱敏，上行/发布前再脱敏一次；模式覆盖 API key/JWT/私钥/口令/DSN/云凭据/设备标识/个人信息/受控路径/`.memoryignore` 区域；脱敏器版本化、可测试。
- 来源与许可分类：`ownership/license/redistribution` 三字段；有用但受限的记录不得晋升出许可范围。
- 删除：私有原始内容物理删除；派生私有对象删除或再生；共享候选撤回/隔离；策展记录以取代/tombstone 为主，隐私/法律/安全要求时物理移除；技能证据失效触发重评。
- 审计：认证与 workspace 注册、检索与返回 id、全文获取、反馈、候选提交、关系与晋升决定、覆盖与策略变更、删除、发布与索引重建——审计不存不必要的原始内容。

### 11.3 治理角色

用户/贡献者、项目记忆策展人、团队知识评审人、技能评测人、安全/隐私管理员、服务运维、发布管理员。**任何会话 agent 都没有直接合并权**。

### 11.4 留存默认

原始本地事件 30 天；本地 episode 90–180 天；显式私有记忆直到删除或策略到期；服务端候选直到裁决或到期；策展记录按治理生命周期；检索审计 180 天。均可按项目密级配置。

### 11.5 降档/可选项（v2.0 修订）

| 项 | v1.0 立场 | v2.0 立场 |
|---|---|---|
| 本地静态加密 | 必需 | **策略项**：受管内网台式机默认关（依赖磁盘级/域策略），笔记本/离网设备默认开 |
| tenant 层级 | 一等公民 | **字段保留、值固定**：内部部署收敛为 org/team 两级；tenant_id 常量占位，未来对外再启用 |
| 法务 hold | 内建 | 移至 L 档需求，S/M 档不实现 |
| 对象存储 | 生产必需 | S 档文件系统；M 档起可选 |
| 消息队列 | 生产必需 | S/M 档进程内或轻量队列；L 档才上 NATS/Kafka |

────────

## 12. Schema 与版本策略（v2.0 新增）

新增 schema 家族：`experience_event`、`session_episode`、`knowledge_candidate`、`retrieval_feedback`、`access_policy`（均 Draft-07、`additionalProperties:false`，与现有七份并列）。

**版本影响量化**：`release.py` 以 schema 内容哈希变化判 major。因此：

1. 新 schema 家族 + 现有 `memory_item` 等的 `source[].kind` 扩展 `chat`/`episode` 合并为**一次 major 发布：hub 1.0.0**；
2. 该发布**不改动**现有七份 schema 的既有字段与语义（仅枚举扩充），现有记录零迁移；
3. 迁移期兼容窗口：`skillhub_*` 别名与旧 resolve 签名保留 ≥2 个 minor；1.2.0 起标记 deprecated，2.0.0 移除；
4. 每个 context pack 记录 `hub_version + hub_commit + resolver_version + index_version`（不可变钉版，取代 `pin: HEAD` 占位）；
5. 事件/episode 信封自带 `schema_version + adapter_version`，服务端可结构化拒绝不支持版本。

────────

## 13. 可靠性语义

`at-least-once 传输 + 幂等应用 + 每会话有序游标 + 以 OpenCode 会话为权威的对账` —— 实用等价于可靠逻辑捕获，不承诺分布式 exactly-once。

- 本地确认：插件仅在本地事务提交后确认；无法提交给出可见警告。
- 服务确认：返回最高连续接受序号；客户端保留未确认数据直到同步成功或策略过期。
- 崩溃恢复：打开 WAL → 校验 schema/加密状态 → 恢复 outbox → 查服务游标 → 对账最近会话 → 修复未配对工具调用 → 无重复地继续。
- 压缩处理：上下文压缩前后存小型续接凭据（当前目标、活动 episode id、context pack id、关键决策、范围内文件/符号、最后本地序号、隐私模式），不回注全量历史。
- 背压：批量压缩、元数据优先于可选 blob、本地配额、队列健康可见、**绝不阻塞交互路径**；配额耗尽时停止确认新捕获并警告，而不是静默丢弃。

指标：本地追加 p95 <10ms；重放逻辑重复 0；已确认事件恢复率 ≥99.99%；断网对 OpenCode 工作影响为零；会话对账收敛到源状态；静默丢失零容忍。

────────

## 14. 分期与人力估算（v2.0 重写）

团队假设：2 名平台工程师 + 1 名兼职评审/维护者；估算为净投入人周（pw），不含评审等待。

**Phase 0（前置，另行跟踪）**：安全整改与既有缺陷修复、base 规则与优化 harness 拆分、`team-assistant` 建立、本地记录规范化适配器、human_decisions 解析、多子系统支持、会话/run 关联、不可变钉版。本文假定其完成。

**Phase 1：本地捕获脊柱 = 强制首切片（S 档，约 6 pw）**
插件事件捕获（会话/消息/工具/空闲）；SQLite WAL + 内容寻址 blob；幂等与有序游标；SDK 会话对账；确定性脱敏；一个本地 Episode/v1；一个本地 KnowledgeCandidate/v1；`memory_context/get` 返回一条真实 Hub 记录（摘录+适用+证据+版本）；`memory_feedback` 闭环；status/private/pause/forget 控制；**无远端原始上传**。
出口：重启/崩溃无已确认事件丢失；重放零重复；对账收敛；断网不影响任务；演示场景（会话 A 产出候选，会话 B 检索复用并反馈）全通。

**Phase 2：检索与用户体验（约 5 pw）**
认证网关（S 档内网令牌）；memory_* 全工具面；紧凑上下文 + 完整出处；一键安装器；本地上下文缓存；特性开关控制的自动检索。
出口：普通会话不选优化管理器即可检索到团队记录；context pack 带不可变版本。

**Phase 3：Episode 与候选蒸馏（约 7 pw）**
目标导向 episode 构建器；确定性+语义两段提取（复用 sediment）；显著性与证据门；脱敏与来源许可分类；个人/项目候选收件箱；增量检查点；候选评审命令流。
出口：标注试点会话产出高精度候选；出处可解析；测试语料零原始转录泄露；失败方法能成为反模式候选；用户纠正压过助手自评。

**Phase 4：协作策展与发布（M 档起，约 7 pw）**
持久化关系与策展决定（七类内核）；PR 自动化；schema 与策略 CI；版本化发布；反馈驱动的排序与退役；删除传播；运维看板。
出口：一个用户的验证经验被另一用户成功复用（§1.5 判据）；冲突被呈现而非覆盖；隔离测试通过。

**Phase 5：技能归纳（约 5 pw，对接现有 GEPA 管线）**
重复过程检测（episode 输入源）→ §6.3 映射的既有管线。
出口：无单 episode 发布的共享技能；每次技能变更有评测证据；合并回退率达标。

合计约 30 pw ≈ 2 人 × 4 个月（S→M 档全量）；只做 Phase 1–2（S 档最小可用）约 11 pw ≈ 2 人 × 6 周。

────────

## 15. 测试与验收

- 单元：事件 id/幂等键、schema 校验、脱敏模式、许可分类、序号与游标、工具配对、episode 边界、显著性谓词、证据加权、**关系分类（48 例基准 + 新增用例）**、ACL、排序特征、留存与删除血缘。
- 插件集成（OpenCode 版本矩阵）：事件载荷形状、消息修订、空闲与压缩、工具生命周期、diff 获取、空闲前崩溃、源对账、启停、磁盘/权限故障。
- 可靠性：同批重放、乱序、丢确认、提交中杀客户端、摄入中杀服务端、损坏 blob、配额耗尽、长离线恢复、schema 版本迁移、半配对恢复。
- 检索：**26 查询基准回归** + 新增标注集（Recall@k/MRR）、多子系统、精确符号/路径、适用与有效期过滤、矛盾呈现、陈旧处理、命名空间与 ACL 隔离、token 预算、compact+exact fetch、无标题幻觉展开。
- 蒸馏：成功构建流、显式失败、无客观测试的用户拒绝、模型称成功但测试失败、同会话重复、同用户重复 episode、他人独立确认、未决结果、混合目标切段、含密钥工具输出、第三方许可材料。
- 安全：跨团队检索、路径注入、伪造身份字段、过期/重放令牌、存储记录内 prompt 注入、恶意贡献者、密钥/PII 语料、删除传播、未授权晋升/遗忘、索引分区泄漏。
- 兼容：`skillhub_*` 行为、既有优化命令与 agent、聚合记忆转换、现有 Hub schema 与发布工具、现有阶段预算与技能依赖闭包、**154 项既有测试全绿**。
- 端到端金路径：capture → reconcile → episode → candidate → review/publish → retrieve → exact fetch → use → validate → feedback。

验收指标沿用 v1.0 §24（可靠性/检索/蒸馏/协作/安全五组），并追加：三个既有基准（154 测试、26 查询、48 例）在每个 Phase 出口保持全绿。

────────

## 16. 风险与对策（修订后）

| 风险 | 对策 |
|---|---|
| 隐私 vs 证据 | 哈希与类型化证据摘要、客观结果制品、显式 opt-in 证据访问 |
| 捕获完备 vs 开销 | 紧凑信封、内容引用、去重、有界留存、对账替代冗余快照 |
| 自动检索 vs 延迟/上下文成本 | 实质性任务判定、小预算、本地缓存、显式回退 |
| 候选量 vs 评审负担 | 强确定性门、证据要求、重复抑制、保守默认共享 |
| 反馈偏差 | 客观证据+显式裁决+独立复用组合；缺反馈不算负面 |
| 投毒与流行度偏差 | 独立贡献者校验、证据加权、冲突检测、策展评审、不以频率推真伪 |
| OpenCode API 漂移 | 版本钉住、兼容矩阵、集成测试、适配器版本化、优雅停用 |
| 运维复杂度 | **规模分档纪律（§4）**：默认 S 档、触发条件升档 |
| 双真相源 | Git/Markdown 对策展知识权威；数据库只存事件/候选/索引/工作流状态 |
| 内核知识过度泛化 | 显式技术范围、配置谓词、版本约束、保守晋升 |
| （新增）复用约束被绕过 | §6 列为实现约束；PR 评审 checklist 增"是否复用既定资产"项 |
| （新增）L 语义再漂移 | §7.2 表为唯一权威；文档 CI 校验术语表引用 |

────────

## 17. 架构决策记录（ADR）

v1.0 的 ADR-001～015 全部保留（插件捕获、本地 outbox 边界、原始内容默认本地、ingest/MCP 分离、Git Hub 为策展真相、五层分离、at-least-once+幂等+对账、compact+exact fetch、四轴独立、轻量默认助手、一记录一文件规范化、知识/技能分治、不可变钉版、存储内容不可信、窄垂直薄片先行）。v2.0 新增：

| ID | 决策 | 理由 |
|---|---|---|
| ADR-016 | 关系分类内核固定为七类判定器，48 例基准为回归门 | 复用已验证资产；11 项清单归并为 7 关系+2 动作+2 结果 |
| ADR-017 | 规模分档 S/M/L，默认 S，升档需触发条件+评审 | 防止平台级组件先于需求落地 |
| ADR-018 | L0–L3 仅适用于 knowledge/skill；event/episode 是层不是级 | 消除与主设计的成熟度语义冲突 |
| ADR-019 | 新 schema 家族随 hub 1.0.0 major 一次性发布 | 符合 release.py 哈希判级；现有记录零迁移 |
| ADR-020 | 模型工具面命名 memory_*，skillhub_* 为兼容别名 | 一次拍板，≥2 minor 弃用窗口 |
| ADR-021 | 技能归纳对接既有 SkillOpt/GEPA/Pareto/eval_gate 管线 | 同一件事不做两套 |
| ADR-022 | 插件落地前允许契约+钩子过渡捕获；落地后契约降级为引导层 | 兼顾一周可用与最终确定性捕获 |

────────

## 18. 待拍板（压缩后）

1. 身份提供方与团队角色映射（S 档内网令牌方案、M 档 SSO 对接）；
2. 哪些项目（若有）允许原始内容上行；
3. 各密级的本地/服务端留存期；
4. episode 构建初期全本地还是部分中心化；
5. 哪些记录类型允许轻量自动批准、哪些永远双评审；
6. 试点用 embedding/检索基础设施（沿用现有离线向量化起步）；
7. fork/临时 worktree 的 workspace 注册方式；
8. 已发布记录的隐私删除在各策略域的处理；
9. 首批支持的 OpenCode 版本；
10. 首批一等公民的构建/测试格式解析器；
11. 试点标注语料（建立检索与蒸馏基线）。

（v1.0 的"Hub skill 投影 schema"与"项目内自动共享"两项已在 §10.7/§11.2 给出默认答案，从待拍板中移除。）

────────

## 附录 A：事件信封示例（要点版）

```json
{
  "schema_version": "experience-envelope/v1",
  "event_id": "0198f0f4-7d3a-7e60-b114-000000000001",
  "sequence": 42,
  "source": {"client": "opencode", "adapter_version": "1.0.0",
             "session_id": "ses_01K...", "message_id": "msg_...", "source_revision": 3},
  "identity": {"team_id": "kernel-performance", "user_id": "usr_pseudonymous"},
  "workspace": {"workspace_id": "ws_01K...", "revision": "full-git-sha", "workflow": "ad-hoc"},
  "event_type": "tool.result",
  "technical_scope": {"paths": ["sysmgr/memmgr/mem/foo.c"], "symbols": ["reclaim_page"],
                      "subsystems": ["memory-management"]},
  "payload": {"summary": "...", "content_ref": "sha256:...", "content_hash": "sha256:..."},
  "memory_context": {"context_pack_id": "ctx_01K...", "hub_version": "1.0.0",
                     "hub_commit": "full-sha", "retrieved_ids": ["F031", "B014"]},
  "policy": {"visibility": "personal", "sensitivity": "internal",
             "sharing_state": "private", "raw_upload_allowed": false,
             "retention_policy": "local-30d", "redaction_version": "3"},
  "integrity": {"idempotency_key": "ses:msg:part:3", "envelope_hash": "sha256:..."}
}
```

Episode/v1、KnowledgeCandidate/v1、memory_context 响应的完整示例沿用 v1.0 附录 A（字段不变，maturity 语义按 §7.2 修正）。

## 附录 B：配置要点

OpenCode 配置：instructions 只载 `base.md`；`default_agent: team-assistant`；`mcp.hm-team-memory` 指向 `{env:HMOPT_MEMORY_MCP_URL}`；`plugin: ["@hmopt/opencode-team-memory"]`；**所有凭据 `{env:...}` 引用**。本地策略（capture include 列表、retention、sharing 默认 private、`auto_publish_to_hub: false`、`.memoryignore`）与 v1.0 附录 B 相同。

## 附录 C：兼容映射

| 现有产物/操作 | 新架构映射 |
|---|---|
| `.opencode/memory/targets|subsystems/*.md`、`global_lessons.md` | 遗留适配器 → 规范一记录一文件本地记录 |
| `human_decisions/*.md` | 专用解析器 → 决策证据 |
| idea ledger / bad-plan state | idea / 反模式候选与状态史 |
| reviews / bench 报告 | 客观证据与 episode 附件 |
| `skillhub_resolve/sediment/status` | `memory_context` / 遗留收割+checkpoint / `memory_status` 的兼容包装 |
| `current_task.json` | 指向每会话关联状态的指针，非持久真相源 |
| Hub skill | 规范技能 → OpenCode 投影（含 name/description/hub-version/content-hash） |

## 附录 D：来源引用

仓库基线、OpenCode plugins/SDK/rules/skills/config/MCP 文档与 MCP 规范链接沿用 v1.0 附录 E；本文新增引用：`docs/Team_Skill_Hub_Design_CN.md` v2.3（七类判定 §10.1.0、成熟度 §4.2、检索 §12）、`src/hmopt/memory/local_curator.py` 与 48 例基准、`hm-skill-hub/tools/`（skill_optimizer/eval_gate/pareto/auto_merge_gate/release/broadcast）、`hm-skill-hub/eval/retrieval/`（26 查询基准）。

────────

## 最终建议

以分层扩展而非替换的方式实施：

```text
OpenCode 插件 → 本地持久 outbox 与对账 → 私有目标导向 episode
  → 证据挂钩候选 → 受治理的 Skill Hub 发布 → 紧凑检索与全文获取 → 验证结果反馈
```

现有优化 pipeline 成为这个通用团队学习平台的一个高级消费者。普通用户以最小工作流开销获得记忆；组织对隐私、证据、晋升与 agent 行为保持严格控制；而每一处新能力都站在已验证资产（七类判定、混合检索、GEPA 评测管线、沉淀链、治理门）之上生长，不另起炉灶。
