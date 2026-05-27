# 团队级 Skill / Memory 仓库闭环方案设计（Team Skill Hub）

| 项 | 值 |
|---|---|
| 文档状态 | Draft v1（待团队评审） |
| 日期 | 2026-05-27 |
| 适用范围 | `.opencode/` harness 的团队化演进；新增独立中央仓 `hm-skill-hub` |
| 语言策略 | 散文 zh-CN；路径 / 字段名 / 代码 / CLI / commit 一律英文（遵循 `.opencode/config.yaml`）|
| 关联文档 | `.opencode/docs/harness_engineer_system.md`、`.opencode/docs/memory_system.md`、`docs/HMOPT_Three_Layer_Architecture_CN.md` |

---

## 0. TL;DR（一页结论）

现有 `.opencode/` 把**两类性质完全不同的资产**混在一个目录里：

- **Skills（过程性指令）**——可被 eval 衡量、可被优化的「程序」。
- **Knowledge（事实 / 记忆）**——不断追加、需去重与冲突消解的「学习到的状态」。

**本方案的脊柱：把这两类拆开，用两套不同的「合并引擎」和「质量门」分别治理，再通过一条带验证门的「沉淀漏斗」把团队每个成员的本地经验汇入一个独立的、语义化版本管理的中央仓 `hm-skill-hub`，并以 pinned 版本反向喂回 pipeline，形成闭环。**

- 过程性 **Skills** → 用 **SkillOpt** 思路治理：技能文档 = 冻结模型的「可训练外部参数」，任何修改必须通过**留出验证集（eval 套件）严格变好才接受**；用 **GEPA Pareto 前沿**解决「多成员互补但互斥的编辑」塌缩问题。
- 事实性 **Knowledge** → 用 **memU / Mem0 / Zep** 思路治理：分层、类型化、稳定 ID、**追加 + 去重 + 冲突消解**（而非 git 行级合并）、双时态保留被取代记录。

四个核心保证：**可迭代**（定时 SkillOpt 优化作业）、**可沉淀**（三层 + L0–L3 成熟度阶梯）、**可闭环**（消费→蒸馏→晋升→eval 发布→再消费）、**稳定可用**（semver + lockfile + eval-gate + 本地兜底 + 脱敏门）。

---

## 1. 背景与问题陈述

### 1.1 现状（as-is）

`.opencode/` 已具备「流程编排 + 记忆沉淀 + 技能复用」的雏形：

- **多 Agent 分阶段流水线**（`docs/harness_engineer_system.md`）：
  `intake → research → plan review(GATE) → implement → code review(GATE) → test(A/B) → decision`，
  hub-and-spoke 委派（仅 `os-opt-manager` 持 `delegate`），每阶段强制 handoff packet，硬门禁不可跳过。
- **Skills = 路径加载的指令包**（非厂商 API），`@`-inline 进上下文后传播给所有子代理。
- **记忆系统已是分层 + 追加 + 去重的雏形**：

  | 存储 | 粒度 | 角色 |
  |---|---|---|
  | `memory/targets/<t>.md` | 每目标 | 稳定结构事实 |
  | `memory/subsystems/<s>.md` | 每子系统 | 跨任务复用知识 |
  | `memory/global_lessons.md` | 全局 | 启发式 / 反模式 |
  | `memory/human_decisions/<slug>.md` | 每目标、按轮次 | 人机决策时间线（实时落盘、抗压缩）|
  | `memory/idea_ledger/<slug>.md` | 每条机制 | 稳定 ID（`L001`，永不复用/删除），状态机 `approved/landed/rejected/reverted/deferred` |
  | `state/bad_plans.md` | 全局/子系统 | 反向去重 |

- **已有闭环意识**：`iterative-optimization.md`（auto-iterate 多轮）、`optimization-funnel.md`（5-想法漏斗 + 强制去重）、`harness-engineer-instruction-count-upgrade_plan.md`（系统已在用自己的流水线优化自己的 harness）。

### 1.2 缺口（gap）

现状距离「团队级可沉淀、可复用、可闭环」还差三件事：

1. **跨成员汇聚机制**——每个成员的沉淀只活在本地 `.opencode/`，无法统一。
2. **统一质量门禁标准**——晋升为 memory 的判据是人工、口头、不可机器校验的。
3. **自动化迭代优化闭环**——技能靠手工改文案，没有 SkillOpt 式的「验证门 + 有界编辑 + 拒绝缓冲」。

### 1.3 关键观察

现有的「idea_ledger 行（追加 / 稳定 ID / 状态机）→ 蒸馏 → global_lessons」这条**本地晋升链**，正是要放大到**跨成员、跨仓库**的那条链。本方案不是推倒重来，而是**把这条链外置、加门、加合并器**。

---

## 2. 设计目标与非目标

### 2.1 目标

| 目标 | 含义 | 主要支撑机制 |
|---|---|---|
| 可迭代 | 技能/知识能随使用持续演进 | 定时 SkillOpt 优化作业（§11）|
| 可沉淀 | 经验能被结构化、分级、固化 | 三层 + L0–L3 成熟度阶梯（§8）|
| 可闭环优化 | 消费→产出→晋升→发布→再消费 | 闭环架构（§5）+ eval-gate（§9）|
| 可复用 | 跨成员、跨子团队、跨工具复用 | SKILL.md 标准 + hub/local 叠加（§12）|
| 稳定性 | 发布不回归、可回滚、可审计 | eval-gate + semver/tag + scorecard（§14）|
| 可用性 | 中央仓故障不阻塞本地优化 | lockfile + vendored 本地兜底（§12.3）|

### 2.2 非目标

- **不**追求把 `.opencode/` 整体搬走——执行面继续贴近代码与任务上下文。
- **不**在第一阶段引入重型基础设施（图数据库集群等）；优先 git + 文件 + 轻量索引。
- **不**做模型微调——沿用 SkillOpt「冻结模型、训练文档」的零推理开销路线。
- **不**追求实时秒级同步——团队知识以「分钟～天」级批量沉淀为节奏。

---

## 3. 业界调研与映射

| 来源 | 核心机制 | 在本方案中的角色 |
|---|---|---|
| **Microsoft SkillOpt**（arXiv 2605.23904, 2026-05）| 技能=冻结模型的可训练外部状态；优化器把打分 rollout 转成**有界 add/delete/replace 编辑**；编辑只有在**留出验证集严格变好**时才接受；稳定性靠「文本学习率预算 + 被拒编辑缓冲 + epoch 慢更新」；产出 `best_skill.md`，跨模型/harness 可迁移 | **Skills 合并引擎 B + 质量门**：CI eval-gate=发布门；有界编辑=每次发布的文本学习率；被拒编辑缓冲=`bad_edits` 注册表；`best_skill.md`=发布制品 |
| **memU**（NevaMind-AI）| 三层 Resource→Item→Category；Memory-as-File-System；自动归类 + 交叉引用成知识图；RAG（快）+ LLM（深）双检索；多智能体 `where` 作用域过滤 | **Knowledge 分层与检索**：Tier 0/1/2；hub（团队）vs local（个人）= `where` 作用域叠加 |
| **Mem0 / Mem0g**（2026-04）| 两阶段：抽取 → 冲突检测 + 图更新；三作用域（user/session/agent）；Conflict Detector 标记重叠 → LLM 消解 | **Curator 合并器**：去重 + 矛盾检测 + LLM 消解 |
| **Zep / Graphiti**（2025-01）| 双时态知识图，事实带「发生时间 + 失效时间」 | **被取代记录不删除**（打 `superseded` + 有效期，可审计）|
| **ExpeL**（arXiv 2308.10144）| 成败经验池 → 抽取跨任务洞见，对洞见做 ADD/UPVOTE/DOWNVOTE/EDIT；正向迁移 | **晋升打分与衰减**：confirmations 多→升权，反例→降权 |
| **GEPA**（arXiv 2507.19457, ICLR'26 Oral）| 反思式 prompt 进化 + **Pareto 前沿**（保留在某实例上最优的一组候选，而非单一全局最优）；用自然语言反馈而非标量奖励 | **避免多成员编辑塌缩**：技能候选保留为 Pareto 前沿，定期合并互补 lesson |
| **Voyager**（arXiv 2305.16291）| 永不停止增长的技能库；入库前**自验证**；按 embedding 检索、可组合 | **技能库范式**：入库前必过验证；组合式复用；抗灾难性遗忘 |
| **Anthropic Agent Skills / SKILL.md**（2025-12 开放标准）| SKILL.md=YAML frontmatter + 指令；plugin=分发、skill=内容；project-scope 经版本控制共享；marketplace 带安全扫描与策展 | **分发与互操作标准 + 治理**：submodule=project-scope 共享；secret-scan/lint CI=marketplace 安全扫描；跨 OpenCode/Claude Code/Codex 复用 |

**研究结论**：SkillOpt 给「技能优化工程化范式」（有界编辑 / 验证门 / 拒绝缓冲 / 慢更新），memU 给「记忆资产组织范式」（层级化 / 结构化 / 可追溯 / 可迁移）；与本系统最契合的做法是——**保留 `.opencode/` 作为执行面，新增 hub 作为组织资产面，用 CI + eval 把「经验沉淀」变成「可验证迭代」**。

---

## 4. 核心设计原则（脊柱）

### 4.1 原则一：两类资产必须分治（全案最重要决定）

| | **Skills（过程性）** | **Knowledge（事实/记忆）** |
|---|---|---|
| 业界对应 | SkillOpt / Anthropic SKILL.md / Voyager | memU / Mem0 / Zep / ExpeL / idea_ledger |
| 基数 | 少、密、可人工审 | 多、稀疏、需索引 |
| 增长方式 | **就地编辑**（add/delete/replace）| **追加 + 合并 + 去重** |
| 质量门 | **eval 套件回归门**（留出验证）| **证据 + 出处 + 去重/冲突消解** |
| 消费方式 | `@`-inline 进上下文 | 按需检索（RAG + ledger 查表）|
| 节奏 | 慢、批量、epoch 式发布 | 持续 |
| 合并引擎 | **引擎 B**：验证门竞争式编辑 + Pareto | **引擎 A**：集合并 + 去重 + 冲突消解 |

> **反模式警告**：用同一套 git 行级合并治理两者，是绝大多数「团队记忆仓」失败的根因——知识会重复/自相矛盾，技能会被某人一周的坏经验覆盖。**两套引擎、两种节奏、两道门，分开走。**

### 4.2 原则二：三层知识层级（memU raw→item→category 的工程化）

- **Tier 0 — 运行轨迹**（本地、易失、每成员）：raw rollout = live session 制品（`current_task.json`、live `human_decisions` 追加、`*_design.md`、`plans/`、`reviews/`、`bench/`）。高量低信噪、可能含设备序列号/密钥。**不直接共享**，是蒸馏输入。
- **Tier 1 — 候选沉淀**（本地→暂存）：从 Tier 0 蒸馏出的**带 schema 的结构化单元**（idea_ledger 行、target/subsystem 笔记、validated delta、bad_plan）。每条带出处 + 证据 + 置信分。**被提议晋升**。
- **Tier 2 — 核心共享资产**（中央仓、团队级）：经验证、去重、合并后的 `best_skill.md` / 策展知识。**反向喂回 pipeline**。

### 4.3 原则三：L0–L3 成熟度阶梯（与三层正交）

成熟度刻画「一条沉淀有多可信、可被多大范围复用」，叠加在三层之上：

| 等级 | 名称 | 判据 | 所在层 | 可见范围 |
|---|---|---|---|---|
| **L0** | draft 草稿 | 仅本地、未结构化 | Tier 0/1 | 仅本人项目 |
| **L1** | candidate 候选 | 结构化完整（字段齐全）+ 初始证据 | Tier 1（staging）| PR 评审中 |
| **L2** | stable 稳定 | 通过团队评审 + eval 套件达标 | Tier 2 | 全团队 |
| **L3** | core 核心 | 跨子团队复用成功 | Tier 2 / `skills/core` | 组织级金标准 |

晋升只能逐级（L0→L1→L2→L3），且每级有明确门（§9）。降级与废弃见 §13.3。

### 4.4 原则四：可追溯优先（provenance-first）

每条 Tier 2 记录必须回链到产生它的 run / 成员 / 证据（commit、review、bench 输出、scorecard）。**无出处 = 不可入库**。这是可信、可审计、可调试「agent 当初为何这么做」的基础。

---

## 5. 总体架构与闭环

```
   ┌────────────────────── hm-skill-hub (Tier 2, 团队共享, semver) ──────────────────────┐
   │                                                                                      │
   │   skills/  (引擎 B: SkillOpt 门 + GEPA Pareto)     knowledge/ (引擎 A: memU 合并)     │
   │      └─ best_skill.md / candidates/ / evals/          └─ 类型化记录 / idea_ledger     │
   │                                                                                      │
   └───────▲─────────────────────────────────────────────────────────────┬──────────────┘
           │ (4) 发布: eval-gate 通过 → 升版本 + tag + scorecard            │ (1) 消费:
           │                                                                │   submodule pin
   (3) 晋升/合并 (Curator-agent + CI)                                       │   + skill-memory.lock
       去重 / 冲突消解 / eval-gate / Pareto-merge / 脱敏                     │   + RAG/ledger 检索
           ▲                                                                ▼
           │                                                   ┌──────────────────────────┐
       staging/ (Tier 1 候选, 类型化 + 证据)                    │  pipeline 运行 (每成员)    │
           ▲                                                   │  manager→research→plan-rev │
           │ (2) 收口点蒸馏 Tier0→1 (hmopt sediment)            │  →code→code-rev→test→dec   │
           │     idea_ledger 行 / target 笔记 /                 └────────────┬─────────────┘
           └───────────────  validated delta / bad plan ◄───────────────────┘ Tier 0 本地沉淀
```

**闭环四步**：

1. **消费**：pipeline 启动时从 hub 拉取 pinned 版本的稳定技能与知识快照（submodule + `skill-memory.lock`）。
2. **蒸馏**：运行中在收口点产出 Tier 0 → Tier 1 候选（`hmopt sediment`）。
3. **晋升/合并**：`hmopt sediment` 把符合条件的候选打包成 PR；Curator-agent + CI 跑去重/冲突/eval/脱敏门，分引擎合并。
4. **发布**：eval-gate 通过则升 hub 版本、打 tag、生成 scorecard 与 release notes；pipeline 重新 pin。

每一环都有门，脏数据不会滚雪球。

---

## 6. 仓库布局

### 6.1 中央技能仓 `hm-skill-hub`（独立 git 仓、semver 打标）

```text
hm-skill-hub/
  README.md
  CONTRIBUTING.md                  # 沉淀协议: 何时/如何贡献
  GOVERNANCE.md                    # CODEOWNERS / 评审规则 / 发布节奏
  registry.yaml                    # 清单: skill 列表 + 版本 + eval 状态 + owner
  CHANGELOG.md                     # 全仓发布日志

  schemas/                         # 每种记录的 JSON-Schema (lint 门)
    memory_item.schema.json
    idea.schema.json
    skill_frontmatter.schema.json
    skill_patch.schema.json
    scorecard.schema.json

  skills/                          # Tier 2 过程性技能 (引擎 B, SKILL.md 标准)
    core/                          # L3 金标准 (跨子团队验证)
      optimization-funnel/
        SKILL.md                   # YAML frontmatter + 正文
        best_skill.md              # 当前验证通过制品 (SkillOpt 产出)
        CHANGELOG.md
        evals/                     # ★本技能的留出验证集 (全案最关键资产)
          cases/*.yaml
          rubric.md
        candidates/                # GEPA Pareto 前沿候选 (待合并互补 lesson)
        scorecards/                # 每版本一张评测卡
      stage-gate-enforcement/
    domain/                        # L2 领域技能
      kernel/
        flash-device-operations/
        ab-test-comparison/
      storage/

  knowledge/                       # Tier 2 策展记忆 (引擎 A, memU/Mem0 分层)
    global/
      lessons/G001-*.md            # global_lessons.md 拆成"每 lesson 一文件"
      anti_patterns/A001-*.md
    subsystems/
      mm/<s>.md
      fs/<s>.md
    targets/
      <target_slug>/
        facts/F001-*.md
        decisions/                 # 来自 human_decisions 的稳定摘要
        idea_ledger.md             # 合并后的权威 ledger (追加, 稳定 ID)
    index/                         # 向量索引清单 (faiss/pgvector) 或重建配方

  evidence/                        # 可验证证据 (防"口头经验")
    benchmarks/
    regressions/

  eval/                            # SkillOpt 式验证门资产
    task_suites/                   # 留出任务集 (按场景组织)
    scorecards/                    # 全局评测卡汇总

  policies/                        # 一等治理文档
    promotion_policy.md            # 何时可沉淀 / 晋升判据
    merge_policy.md                # 两套合并引擎规则
    deprecation_policy.md          # 废弃与失效治理

  staging/                         # Tier 1 入站候选 (成员 PR 落区, 策展前)
    <member>/<date>/*.json

  tools/                           # 沉淀/合并/评测 CLI + curator 提示词
    sediment.py
    merge_curator.md               # Curator-agent 提示词
    run_evals.py
    lint.py
    dedup.py
    redact.py                      # 脱敏

  releases/
    skill-memory-<semver>/         # 发布快照 (供 lock 引用)

  .github/workflows/               # CI: lint + secret-scan + eval-gate + index-build
    ci.yml
```

**为何把 `global_lessons.md` / ledger 拆成「每条一文件 / 稳定 ID 一行」？** 这是让合并可行的关键：不同 ID = 不同文件/不同行 ⇒ git 行级冲突几乎消失，合并退化为「集合并」，真正的去重/冲突交给 Curator 语义处理（§10.1）。

### 6.2 消费端：`.opencode/` 瘦身为「共享 + 本地」叠加

```text
.opencode/
  hub/                  # git submodule (或 vendored release), pin 到 hm-skill-hub 某版本 (运行时只读)
  local/                # 本成员 Tier 0/1 工作记忆 (gitignore 或个人分支), live 沉淀落这里
    sediment_staging/   # hmopt sediment 产出的待提交候选包
  skill-memory.lock     # 锁定 hub 版本 (semver + commit SHA), 防漂移, 保可复现
  resolver.py           # 加载技能/查记忆时: 先 hub(共享) 再叠加 local(个人在途)
```

这正是 memU 的 `where` 作用域（team vs personal）与 Anthropic 的 project-scope vs personal-scope。**可用性保证**见 §12.3。

---

## 7. 数据模型与 Schema

### 7.1 类型化记忆体系（memory item types）

每条 Knowledge 记录必须声明 `type`，五类之一：

| type | 含义 | 示例 |
|---|---|---|
| `fact` | 稳定结构事实 | "`hyperhold_write_eid` 在 iotab.c:312，热路径每次写 16B eid" |
| `rule` | 操作规则 | "改 `process_one_work` 前必须确认 wq lock 跨调用语义" |
| `pattern` | 可复用正向模式 | "把 loop-invariant 锁检查 hoist 出热循环" |
| `anti_pattern` | 反模式 | "对 kworker 入口盲目 inline 会撑爆 i-cache（phone X 实测回归）" |
| `playbook_step` | 流程步骤片段 | "A/B 测试：先 flash stock 跑测，再 flash feature 跑测，再比对" |

### 7.2 `memory_item.schema.json`（核心 schema 草案）

```json
{
  "$id": "memory_item.schema.json",
  "type": "object",
  "required": ["id", "type", "title", "body", "scope", "source", "created_at", "maturity"],
  "properties": {
    "id":        {"type": "string", "pattern": "^[FGAR][0-9]{3,}$"},
    "type":      {"enum": ["fact", "rule", "pattern", "anti_pattern", "playbook_step"]},
    "title":     {"type": "string", "maxLength": 120},
    "body":      {"type": "string"},
    "scope":     {"type": "object", "required": ["level"],
                  "properties": {
                    "level":      {"enum": ["function", "call-site", "data-flow", "subsystem", "architectural", "global"]},
                    "subsystem":  {"type": "string"},
                    "target_slug":{"type": "string"}}},
    "source":    {"type": "array", "minItems": 1,
                  "items": {"type": "object",
                    "required": ["kind", "ref"],
                    "properties": {
                      "kind": {"enum": ["commit", "review", "bench", "doc", "run_id"]},
                      "ref":  {"type": "string"}}}},
    "evidence":  {"type": "object",
                  "properties": {
                    "delta_pct":     {"type": "number"},
                    "compare_level": {"enum": ["total", "process", "thread", "lib", "function"]},
                    "confirmations": {"type": "integer", "minimum": 1}}},
    "maturity":  {"enum": ["L0", "L1", "L2", "L3"]},
    "status":    {"enum": ["active", "superseded", "deprecated"]},
    "invalidation": {"type": "string",
                     "description": "失效条件, e.g. 'kernel rebase 后须重校 offset'"},
    "supersedes":   {"type": "array", "items": {"type": "string"}},
    "valid_from":   {"type": "string", "format": "date-time"},
    "valid_until":  {"type": "string", "format": "date-time"},
    "score":        {"type": "number"},
    "contributor":  {"type": "string"},
    "created_at":   {"type": "string", "format": "date-time"},
    "updated_at":   {"type": "string", "format": "date-time"}
  }
}
```

ID 前缀约定：`F`=fact、`G`=global lesson、`A`=anti_pattern、`R`=rule/playbook。沿用 idea_ledger 的「稳定 ID、永不复用、永不删除」纪律。

### 7.3 `skill_frontmatter.schema.json` 与「技能更新清单」

技能 `SKILL.md` 的 YAML frontmatter（兼容 Anthropic SKILL.md 标准）：

```yaml
---
name: optimization-funnel
version: 2.3.1                 # semver
scope: [kernel, generic]
maturity: L3
eval_id: eval/task_suites/funnel_suite_v2
owners: [@kernel-perf-team]
status: active
---
```

**每次技能更新（PR）必须附「更新清单」**（`skill_patch.schema.json`），强制绑定三元组：

```json
{
  "skill": "optimization-funnel",
  "edit_ops": [{"op": "replace", "anchor": "## Ranking Questions", "rationale": "..."}],
  "task_suite": "eval/task_suites/funnel_suite_v2",
  "metrics": {"pass_rate": 0.0, "instr_count_delta": 0.0, "regression_rate": 0.0},
  "baseline_version": "2.3.0",
  "rejected_buffer_ref": "skills/.../bad_edits.jsonl"
}
```

无 `task_suite` + `metrics` 的技能 PR 直接被 CI 拒绝。

### 7.4 idea_ledger（沿用现有结构，仅外置 + 合并）

现有 `.opencode/memory/idea_ledger/template.md` 的结构（`id / mechanism / scope / status / verdicted_by / delta_pct / validation_path / reopen_trigger …`）**完整保留**，只是：① 字段固化为 JSON（`idea.schema.json`）便于机器合并；② 从本地外置到 `hub/knowledge/targets/<slug>/idea_ledger.md`；③ 由 Curator 做跨成员合并（§10.1）。

---

## 8. 沉淀时机与成熟度晋升

### 8.1 Tier 0 → Tier 1（蒸馏）— 何时触发

复用系统已有的自然收口点（现 `memory-accumulation.md` 已在做，本方案只固化输出 schema）：

- pipeline 的 **decision 阶段**结束；
- 人机会话（`kernel-plan` / `kernel-research`）**"done"**；
- 每个 **auto-iterate pass** 末（`iterative-optimization.md`）。

产物：`.opencode/local/sediment_staging/*.json`，每条符合 §7 schema，标 `maturity: L1`。

### 8.2 Tier 1 → Tier 2（晋升）— 促发条件（Promotion Triggers）

**不是每次 run 都晋升**。一条候选可被提议晋升，当且仅当满足以下之一：

1. 在 **≥2 个独立任务**中复现收益；
2. **单任务收益显著**且有 bench 证据（如指令数/时延/正确率改善，附 `validation_path`）；
3. **失败教训**具高复用价值（可防止重复踩坑 → `anti_pattern` / bad_plan）。

且必须同时通过 §9 的三道门。

### 8.3 贡献节奏

推荐 **自动暂存（持续、廉价）+ 批量 PR（每周或里程碑）**：`hmopt sediment` 把符合条件的 Tier 1 候选打包成单个「沉淀 PR」投向 hub —— 批量便于统一去重、避免 PR 刷屏。

### 8.4 成熟度晋升路径

```
L0 (本地草稿) ──结构化+初始证据──▶ L1 (候选, 入 staging/)
L1 ──团队评审 + eval 达标──▶ L2 (stable, 入 knowledge/ 或 skills/domain/)
L2 ──跨子团队复用成功 (≥2 子团队 / ≥N 次正向引用)──▶ L3 (core, 入 skills/core/)
```

---

## 9. 质量保证体系（三道门）

```
候选(L1) ──▶ [门1: Schema/Lint/脱敏] ──▶ [门2: 证据门] ──▶ [门3: 策展/评审 + eval] ──▶ 稳定(L2/L3)
              廉价/自动/CI                  自动校验           Curator-agent + 人 + eval-gate
```

### 9.1 门 1：Schema / Lint / 脱敏（廉价、自动、CI/pre-commit）

- **Schema 校验**：每条记录/技能过 §7 JSON-Schema（必填字段、出处、type 合法）。畸形即拒。
- **脱敏门（★它方案缺失，必须有）**：内核 flash/测试场景会带**设备序列号、签名 key**。`tools/redact.py` + CI `secret-scan` 强制扫描；命中即拒并 `[REDACTED]`（复用 `human-interaction-memory.md` 既有脱敏规则）。团队仓泄密是放大事故，此门不可省。

### 9.2 门 2：证据门（自动）

- 知识声明需引用（`validation_path` / `delta_pct` / `confirmations ≥ N`）；技能编辑需 eval 结果。
- 无证据 → 留在 L1（候选），不得进 L2。（ExpeL 成败接地 + SkillOpt 留出验证）

### 9.3 门 3：策展 + eval（Curator-agent + 人 + eval-gate）

- **去重 / 冲突 / 范围 / 泛化性**由 Curator-agent 预处理（§10.1），再由**双评审人**签字：1 名**领域 reviewer**（结论对不对）+ 1 名**流程 reviewer**（是否合规、可复用）。
- 技能类额外过 **eval-gate**：候选技能在留出 task suite 上 A/B，**严格变好**才可合入（§10.2）。
- **不设"显式豁免"口子**（★修正它方案的风险点）：要破例只能「降级为 L1 候选 + owner 签字 + 下周期复核」，不得直接合入生产。

### 9.4 打分函数（排序 + 衰减）

```
score = w1·evidence_strength      # delta 幅度 / bench 质量
      + w2·confirmations          # 独立复现次数 (ExpeL UPVOTE)
      + w3·recency                # 新近度
      + w4·generality             # scope 越宽越高 (subsystem/architectural > function)
      - w5·counter_evidence       # 反例 (ExpeL DOWNVOTE)
      - w6·staleness              # invalidation 触发 (如 rebase) 后衰减
```

score 用于晋升排序、检索排序与废弃判定。

---

## 10. 合并机制（两套引擎）

### 10.1 引擎 A — Knowledge（追加型）：集合并 + 去重 + 冲突消解

**绝不用 git 行级合并。** 每条记录是不可变、内容寻址、带稳定 ID 的单元，合并 = 集合并。Curator-agent（Mem0g Conflict Detector + LLM 消解的落地）在 PR 上运行：

```
def curate_knowledge(incoming_items, hub_items):
    for item in incoming_items:
        # 1) 去重: embedding 相似度聚类 (memU/Mem0)
        dup = find_near_duplicate(item, hub_items, threshold=0.92)
        if dup:
            merge_provenance(dup, item)        # 合并出处, confirmations += 1, 重算 score
            continue
        # 2) 矛盾检测: 同一 (target, mechanism) 断言相反
        conflict = find_contradiction(item, hub_items)
        if conflict:
            # 3) 消解: 证据/新近度加权 (Zep 双时态)
            if stronger_evidence(item, conflict):
                conflict.status = "superseded"   # 不删除!
                conflict.valid_until = now()
                item.supersedes = [conflict.id]
                add(item)
            elif high_risk(conflict):
                escalate_to_human(item, conflict)  # 高风险升级
            else:
                drop_with_citation(item, conflict)
        else:
            add(item)
```

**CRDT 式纪律**：追加 + tombstone（状态位 `active/superseded/deprecated`）而非删除——沿用 ledger 现有规则。双时态（`valid_from/valid_until`）接住「内核 rebase 使 offset 事实失效」。

### 10.2 引擎 B — Skills（编辑型）：SkillOpt 验证门 + GEPA Pareto

**绝不用集合并。** 某成员经验提议的技能改动 = 一个**编辑候选**（有界 add/delete/replace），只有在 eval 套件上**严格变好**才接受：

```
def merge_skill_edit(skill, edit, eval_suite, pareto_frontier, bad_edits):
    if edit in bad_edits:                       # 被拒编辑缓冲 (SkillOpt)
        return REJECT("known-bad edit")
    edit = clip_to_budget(edit, textual_lr)     # 文本学习率: 有界编辑预算
    candidate = apply(skill, edit)
    score = run_evals(candidate, eval_suite)    # 留出验证集
    if score.strictly_better_than(skill.score): # 严格变好才接受
        skill = candidate                        # 更新 best_skill.md
        write_scorecard(skill, score)
    else:
        bad_edits.append(edit)                   # 进拒绝缓冲
    # GEPA Pareto: 保留"在某些 eval 实例上最优"的候选 (★它方案缺失)
    pareto_frontier = update_pareto(pareto_frontier, candidate, per_instance_scores)
    return skill, pareto_frontier
```

**为何需要 Pareto（★核心）**：当 N 个成员各提编辑时，单一全局 eval 分会让「互补但互斥」的编辑塌缩到一个局部最优。GEPA Pareto 前沿保留「各自在某些实例上最优」的一组候选（放 `skills/<n>/candidates/`），定期合并互补 lesson。**这才是「团队人人沉淀、统一汇入而不互相覆盖」的正解。**

- **文本学习率** = 每次发布的有界编辑预算（别让某人一周的坏经验重写整篇技能）。
- **epoch 慢更新** = 按发布周期批量合并，而非每 PR 即时重写。

### 10.3 一句话区分

> **知识靠「集合并 + 去重 + 冲突消解」合并；技能靠「验证门竞争式编辑（SkillOpt）+ Pareto（GEPA）」合并。** 两类资产，两台引擎。

---

## 11. 闭环优化流水线（定时作业）

新增 nightly/weekly「Skill/Memory 优化作业」（`tools/` + CI 调度）：

```
(1) Collect    聚合各项目 .opencode/local/sediment_staging 候选 → staging/
(2) Normalize  按 §7 schema 标准化字段 + 去噪 + 脱敏
(3) Cluster    embedding 聚类相似经验 (引擎 A 去重前置)
(4) Optimize   对 skills/*.md 跑 SkillOpt 有界编辑 (引擎 B), 早期半自动: 自动提 PR, 人工合并
(5) Validate   在 held-out task suites 跑 A/B, 出 scorecard
(6) Promote    仅提升收益版本, 升 semver + 打 tag, 更新 registry.yaml
(7) Broadcast  自动生成 release notes, 供业务仓 pin 新版本
```

**早期安全约束**：第 (4) 步的自动优化必须接入 `bad_edits` 拒绝缓冲 + Pareto + 脱敏，且**默认半自动**（自动提 PR、人工合并），积累信任后再逐步放开为全自动。

---

## 12. 消费与集成

### 12.1 版本锁定（`skill-memory.lock`）

业务仓通过 `skill-memory.lock`（semver + commit SHA）固定 hub 版本，等价 package lockfile，做可复现 + 防漂移。运行中的技能不会在脚下悄悄变（SkillOpt `best_skill.md` 是发布制品）。每次 run 记录消费了哪个 hub 版本。

### 12.2 hub/local 运行时叠加（resolver）

`.opencode/resolver.py` 加载技能/查记忆时：**先读 hub（共享）再叠加 local（个人在途）**。这支持「成员使用中沉淀自己的」——本地在途记忆叠加在共享之上，互不污染（memU `where` 作用域）。

### 12.3 可用性：故障优雅降级（★它方案缺失）

- hub 以 submodule pin 在本地（vendored 副本），**中央仓宕机绝不阻塞**本地优化工作。
- resolver 检测 hub 不可达时，回退到上次成功 pin 的本地快照并告警，不中断 pipeline。

### 12.4 跨工具复用

技能采用 Anthropic SKILL.md 开放标准，可同时被 OpenCode / Claude Code / Codex 消费，hub 即「团队私有 skill marketplace」。

---

## 13. 治理与策略

### 13.1 `promotion_policy.md`

固化 §8.2 触发条件 + §9 三道门 + §8.4 晋升路径。明确「谁能提议、谁能晋升」。

### 13.2 `merge_policy.md`

固化 §10 两套引擎规则 + 双评审人（领域 + 流程）+ 无豁免原则 + CODEOWNERS（`skills/core/` 需 owner 团队批准）。

### 13.3 `deprecation_policy.md`（失效治理）

- **废弃状态**：`active → superseded`（被更优记录取代）/ `→ deprecated`（失效条件触发）。
- **触发**：`invalidation` 命中（如 kernel rebase）、score 衰减低于阈值、连续反例。
- **定期清理作业**：标 `deprecated` 的记录从检索/inline 中排除，但**保留可审计**（不物理删除）。

### 13.4 发布节奏

每周小版本（patch/minor）、每月稳定版（minor/major）。`skills/core/` 变更走更严评审。

---

## 14. 稳定性与可用性保障（汇总）

| 保障 | 机制 |
|---|---|
| 可回滚 | 每次技能更新有 tag + 对应 scorecard；`git revert` 即回退 |
| 抗回归 | **CI eval-gate**：eval 不达标禁止发布（让反喂安全的根本机制）|
| 抗破坏性改写 | 文本学习率（有界编辑）+ 被拒编辑缓冲 + epoch 慢更新（SkillOpt 三件套）|
| 可审计 | 每条记录回链来源 + reviewer + scorecard（§4.4）|
| 抗污染 | 候选层（staging/L1）与稳定层（L2/L3）物理隔离 |
| 多版本共存 | 不同项目可 pin 不同版本，逐步升级 |
| 高可用 | lockfile + vendored 本地兜底 + resolver 降级（§12.3）|
| 失效治理 | deprecation 状态 + 定期清理 + 双时态有效期（§13.3）|
| 防泄密 | 脱敏门 + CI secret-scan（§9.1）|

---

## 15. 分阶段落地路线图

| 阶段 | 周期 | 目标 | 交付物 | 风险 |
|---|---|---|---|---|
| **Phase 0｜抽取** | 1–2w | 零行为变更地双仓接口跑通 | 把 `skills/agents/pipelines/commands/docs` 切到 `hm-skill-hub`，submodule 接回并 pin；建仓骨架 + schemas + `registry.yaml` + PR 模板 | 低 |
| **Phase 1｜蒸馏** | 2–3w | Tier0→Tier1 结构化 | `hmopt sediment`（pipeline 末输出候选包）；`memory export` 把现有 memory/plans/reviews 转标准对象 | 低 |
| **Phase 2｜策展+合并** | 3–6w | 知识合并上线（引擎 A）| Curator-agent + lint/secret-scan/dedup CI；人工审批基线；`policies/` 三文档 | 中 |
| **Phase 3｜eval 门** ★ | 6–10w | 安全反喂（引擎 B）| **建 core task suite**（最难长杆）+ CI eval-gate + scorecard；半自动 bounded-edit 优化器 | **高** |
| **Phase 4｜自动优化** | 10w+ | 闭环自动迭代 | nightly/weekly 优化作业（§11）；每周小版本 / 每月稳定版；业务仓 `skill-memory.lock` 防漂移 | 中 |

---

## 16. 风险与缓解

| 风险 | 说明 | 缓解 |
|---|---|---|
| **eval 套件是长杆** ★ | 内核优化 ground truth = 真机 A/B 指令数 delta（慢/贵/噪声大）；没它技能合并退化成「凭感觉」 | Phase 3 重点投入；先用代理指标（编译期静态指令数估计 + 小样本真机）起步，逐步加密真机验证；诚实标注这是关键路径 |
| 过度沉淀（噪声）vs 不足（陈旧）| 沉淀太多变噪声，太少变陈旧 | 打分 + 衰减 + N 次确认阈值调；L0–L3 阶梯过滤 |
| 密钥泄露 | 设备序列号/key 入库 | 脱敏门 + CI secret-scan（强制）|
| 热点文件合并争用 | 多人改同一 memory 文件 | 每记录一文件 + 稳定 ID，规避行级合并 |
| 两类资产混用一台引擎 | 知识自相矛盾 / 技能被覆盖 | §4.1 强制分治 |
| 多成员编辑塌缩 | 单一全局 eval 分丢失互补策略 | GEPA Pareto 前沿（§10.2）|
| eval「豁免」漏洞 | 未验证编辑混入生产 | 无豁免原则；破例只能降级为候选 + owner 签字 + 复核（§9.3）|
| 自动优化过激 | nightly 自动改技能失控 | 早期半自动 + 拒绝缓冲 + Pareto + 人工合并（§11）|
| submodule 摩擦 | 子模块更新繁琐 | 提供 vendored-release 兜底 + lockfile |

---

## 17. 附录

### 17.1 关键 CLI（拟新增到 `hmopt`）

```bash
hmopt sediment                       # 收口点: Tier0→Tier1 蒸馏, 产出候选包到 local/sediment_staging
hmopt sediment --bundle --open-pr    # 打包符合条件候选 → 提 PR 到 hub
hmopt skill-lock --update <semver>   # 更新 skill-memory.lock 到指定 hub 版本
hmopt skill-eval <skill> --suite <s> # 本地跑技能 eval (A/B), 出 scorecard
hub: tools/run_evals.py / dedup.py / merge_curator.md  # hub 侧 CI 调用
```

### 17.2 示例记录（`anti_pattern`）

```yaml
id: A007
type: anti_pattern
title: 盲目 inline kworker 入口撑爆 i-cache
body: 对 process_one_work 等 kworker 热入口做无差别 inline, 在 phone X 实测 i-cache miss 上升, 净回归。
scope: {level: function, subsystem: workqueue, target_slug: wq_threadpool}
source: [{kind: bench, ref: .opencode/bench/wq_threadpool_validation.md},
         {kind: review, ref: reviews/wq_threadpool__iter2_code_review.md}]
evidence: {delta_pct: 1.4, compare_level: function, confirmations: 2}
maturity: L2
status: active
invalidation: "i-cache 容量翻倍的新平台需重测"
contributor: "@dev-a"
created_at: "2026-05-20T10:00:00Z"
```

### 17.3 术语表

| 术语 | 含义 |
|---|---|
| Tier 0/1/2 | 运行轨迹 / 候选沉淀 / 核心共享资产（memU raw→item→category）|
| L0–L3 | 成熟度：draft / candidate / stable / core |
| 引擎 A / B | 知识合并（集合并+去重+冲突）/ 技能合并（SkillOpt 门 + GEPA Pareto）|
| 文本学习率 | 每次发布的有界编辑预算（SkillOpt）|
| 被拒编辑缓冲 | 已验证为坏的编辑，不再重试（SkillOpt）|
| Pareto 前沿 | 「各自在某些 eval 实例上最优」的一组技能候选（GEPA）|
| eval-gate | 发布前的留出验证集回归门——让反喂安全的根本机制 |
| 双时态 | 记录带 valid_from/valid_until，被取代不删除（Zep）|

---

## 18. 待团队拍板的开放问题

1. hub 仓命名与归属（`hm-skill-hub` vs `org/opencode-skill-memory`）、放在哪个 GitHub org。
2. eval task suite 的 ground truth 策略：纯真机 A/B vs 静态代理指标 vs 混合（直接影响 Phase 3 工期）。
3. 沉淀贡献节奏：每周批量 vs 里程碑触发 vs 持续自动。
4. 检索后端：起步用 faiss 本地文件，还是直接上 pgvector（与现有 `storage/` 对齐）。
5. `skills/core` 的 owner 团队与晋升评审人选。
