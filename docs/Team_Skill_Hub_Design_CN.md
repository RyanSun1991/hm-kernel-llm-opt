# 团队级 Skill / Memory 仓库闭环方案设计（Team Skill Hub）

| 项 | 值 |
|---|---|
| 文档状态 | Draft v2.0（待团队评审） |
| 日期 | 2026-05-27 |
| 适用范围 | `.opencode/` harness 的团队化演进；新增独立中央仓 `hm-skill-hub` |
| 语言策略 | 散文 zh-CN；路径 / 字段名 / 代码 / CLI / commit 一律英文 |
| 关联文档 | `.opencode/docs/harness_engineer_system.md`、`.opencode/docs/memory_system.md` |
| 图示 | `docs/Team_Skill_Hub_Design_Diagrams_CN.md`（6 张 Mermaid：闭环 / 双引擎 / 沉淀漏斗 / skills 布局 / 运行时组合 / 路线图）|
| 修订 | v2.0：新增 §6.2「skills/ 内部布局」；中后段整体精简（删比对脚注、压缩 schema、合并治理/稳定/可用） |

---

## 0. TL;DR

现有 `.opencode/` 把**两类性质不同的资产**混在一起：

- **Skills（过程性指令）**——可被 eval 衡量、可被优化的「程序」。
- **Knowledge（事实/记忆）**——不断追加、需去重与冲突消解的「学习到的状态」。

**方案脊柱**：把两类拆开、用两套合并引擎分治，再通过一条带验证门的「沉淀漏斗」把每个成员的本地经验汇入一个独立、semver 版本化的中央仓 `hm-skill-hub`，并以 pinned 版本反向喂回 pipeline，形成闭环。

- **Skills** → 用 **SkillOpt** 治理：技能文档 = 冻结模型的「可训练外部参数」，改动必须在留出 eval 套件上**严格变好**才接受；用 **GEPA Pareto** 解决多成员编辑塌缩。
- **Knowledge** → 用 **memU/Mem0/Zep** 治理：分层、类型化、稳定 ID、**追加+去重+冲突消解**（非 git 行级合并）、双时态保留被取代记录。

四个保证：**可迭代**（定时优化作业）、**可沉淀**（三层 + L0–L3）、**可闭环**（消费→蒸馏→晋升→发布→再消费）、**稳定可用**（semver + lockfile + eval-gate + 本地兜底 + 脱敏门）。

---

## 1. 背景与缺口

`.opencode/` 已有「流程编排 + 记忆沉淀 + 技能复用」的雏形：分阶段流水线（`research → plan review → implement → code review → test → decision`，硬门禁）、路径加载的指令包（skills）、分层记忆：

| 存储 | 角色 |
|---|---|
| `memory/targets/` `memory/subsystems/` | 结构事实 |
| `memory/global_lessons.md` | 启发式 / 反模式 |
| `memory/human_decisions/` | 人机决策时间线（实时落盘）|
| `memory/idea_ledger/` | 每机制裁决，稳定 ID（`L001`），状态机 `approved/landed/rejected/...` |
| `state/bad_plans.md` | 反向去重 |

**距离团队级闭环还差三件事**：① 跨成员汇聚机制（沉淀只活在本地）；② 统一的、可机器校验的质量门；③ 自动化迭代（技能靠手工改文案，无验证门）。

**关键观察**：现有「idea_ledger 行 → 蒸馏 → global_lessons」这条本地晋升链，正是要放大到跨成员、跨仓库的那条链。本方案是把它**外置 + 加门 + 加合并器**，而非推倒重来。

---

## 2. 目标与非目标

| 目标 | 主要支撑 |
|---|---|
| 可迭代 | 定时优化作业（§11）|
| 可沉淀 | 三层 + L0–L3（§4）|
| 可闭环 | 闭环架构（§5）+ eval-gate（§9）|
| 可复用 | SKILL.md 标准 + hub/local 叠加（§12）|
| 稳定可用 | eval-gate + semver + lockfile + 本地兜底（§13）|

**非目标**：不整体搬走 `.opencode/`（执行面贴近代码）；首期不引入重型基础设施（优先 git + 文件 + 轻量索引）；不做模型微调（沿用 SkillOpt 零推理开销）；不追实时同步（分钟～天级批量）。

---

## 3. 业界调研映射

| 来源 | 核心机制 | 角色 |
|---|---|---|
| **SkillOpt**（arXiv 2605.23904）| 技能=可训练外部状态；有界 add/delete/replace 编辑；留出验证集严格变好才接受；文本学习率 + 被拒编辑缓冲 + 慢更新；产出 `best_skill.md` | Skills 引擎 B + 质量门 |
| **memU** | 三层 Resource→Item→Category；Memory-as-File-System；RAG + LLM 双检索；`where` 作用域 | Knowledge 分层与检索；hub/local 叠加 |
| **Mem0 / Mem0g** | 两阶段抽取→冲突检测+消解；Conflict Detector → LLM 消解 | Curator 合并器 |
| **Zep / Graphiti** | 双时态，事实带失效时间 | 被取代记录不删除（`superseded`）|
| **ExpeL** | 成败池 → 抽取洞见 ADD/UPVOTE/DOWNVOTE/EDIT | 晋升打分与衰减 |
| **GEPA** | 反思式进化 + Pareto 前沿（保留互补候选，非单一最优）| 避免多成员编辑塌缩 |
| **Voyager** | 增长式技能库，入库前自验证，可组合 | 技能库范式 + 组合复用 |
| **Anthropic Agent Skills** | SKILL.md 开放标准；plugin 分发；project-scope 版本控制共享；marketplace 安全扫描 | 分发与互操作标准 + 治理 |

**结论**：SkillOpt 给「技能优化工程化范式」，memU 给「记忆资产组织范式」；最契合做法——保留 `.opencode/` 为执行面，新增 hub 为资产面，用 CI + eval 把「经验沉淀」变成「可验证迭代」。

---

## 4. 核心设计原则（脊柱）

### 4.1 两类资产分治（最重要决定）

| | **Skills（过程性）** | **Knowledge（事实/记忆）** |
|---|---|---|
| 增长 | 就地编辑（add/delete/replace）| 追加 + 合并 + 去重 |
| 质量门 | eval 套件回归门 | 证据 + 出处 + 冲突消解 |
| 消费 | `@`-inline 进上下文 | 按需检索（RAG + ledger）|
| 合并引擎 | **B**：验证门竞争式编辑 + Pareto | **A**：集合并 + 去重 + 冲突消解 |

> **反模式**：用同一套 git 行级合并治理两者——知识会重复/自相矛盾，技能会被某人一周的坏经验覆盖。**两套引擎、两道门，分开走。**

### 4.2 三层 + L0–L3 成熟度

三层（memU raw→item→category 的工程化）：

- **Tier 0 运行轨迹**（本地、每成员）：live session 制品（`current_task.json`、`*_design.md`、`plans/reviews/bench`）。低信噪、可能含密钥，**不直接共享**，是蒸馏输入。
- **Tier 1 候选沉淀**（本地→staging）：从 Tier 0 蒸馏出的带 schema 结构化单元，附出处+证据+置信分，**被提议晋升**。
- **Tier 2 核心共享**（中央仓）：经验证、去重、合并后的 `best_skill.md` / 策展知识，**反向喂回**。

成熟度（与三层正交，刻画「多可信、可复用多广」）：

| 等级 | 判据 | 范围 |
|---|---|---|
| **L0** draft | 本地草稿 | 本人项目 |
| **L1** candidate | 结构化完整 + 初始证据 | PR 评审中 |
| **L2** stable | 团队评审 + eval 达标 | 全团队 |
| **L3** core | 跨子团队复用成功 | 组织金标准（`skills/core/`）|

晋升逐级（L0→L1→L2→L3），每级有门（§9）。

### 4.3 三原型轴：每个产物的唯一归宿

「两类资产」落到目录时展开成**三个原型**——决定每个文件去哪的唯一一条轴：

> **它是「程序」（过程、团队共享、可被 eval 优化），还是某次「运行的产物」（证据/轨迹、个人本地）？**

| 原型 | 含义 | 归宿 |
|---|---|---|
| **procedural-shared** | agents / skills / commands / pipelines / harness 规范 / `*_template.md` | **hub** |
| **knowledge-curated** | facts / rules / patterns / anti_patterns / idea_ledger | **hub** `knowledge/`（只放蒸馏精华）|
| **run-evidence-local** | plans / reviews / bench / patches / 每次 design / current_task | **业务仓** `local/`；纯运行态 gitignore |

互斥且穷尽：`.opencode/` 每个文件恰好落入其一（详见 §6.4）。**关键推论**：`agents/` 与 `skills/` 同属 procedural-shared，一起进 hub；`bench/`/`reviews/` 属 run-evidence-local，留业务仓，只有蒸馏精华晋升 hub。

### 4.4 可追溯优先

每条 Tier 2 记录必须回链产生它的 run/成员/证据。**无出处 = 不可入库**。

---

## 5. 总体架构与闭环

```
   ┌──────────────── hm-skill-hub (Tier 2, 团队共享, semver) ────────────────┐
   │   skills/ (引擎 B: SkillOpt + Pareto)     knowledge/ (引擎 A: memU 合并) │
   └──────▲────────────────────────────────────────────────┬───────────────┘
          │ (4) 发布: eval-gate 通过 → 升版本+tag+scorecard   │ (1) 消费: submodule pin
          │                                                   │   + skill-memory.lock + 检索
   (3) 晋升/合并 (Curator + CI)                               ▼
      去重/冲突/eval/Pareto/脱敏                  ┌──────────────────────────┐
          ▲                                       │  pipeline 运行 (每成员)    │
      staging/ (Tier 1 候选)                       └────────────┬─────────────┘
          ▲                                                     │ (2) 收口点蒸馏 Tier0→1
          └──────────  validated delta / 反模式 / ledger 行 ◄───┘
```

1. **消费**：pipeline 启动从 hub 拉 pinned 版本（submodule + lock）。
2. **蒸馏**：运行中在收口点产出 Tier 0→1 候选（`hmopt sediment`）。
3. **晋升/合并**：候选打包成 PR；Curator + CI 跑去重/冲突/eval/脱敏，分引擎合并。
4. **发布**：eval-gate 通过则升版本、打 tag、出 scorecard；pipeline 重新 pin。

每环有门，脏数据不滚雪球。

---

## 6. 仓库布局

### 6.1 中央仓 `hm-skill-hub`（semver 打标）

```text
hm-skill-hub/
  registry.yaml                # 清单: skill 列表 + 版本 + eval 状态 + owner
  schemas/                     # 每种记录的 JSON-Schema (lint 门)
  skills/                      # Tier 2 过程性技能 (引擎 B) —— 内部布局见 §6.2
  knowledge/                   # Tier 2 策展记忆 (引擎 A, memU/Mem0 分层)
    global/lessons/  global/anti_patterns/
    subsystems/<s>.md
    targets/<slug>/{facts/, decisions/, idea_ledger.md}
    index/                     # 向量索引清单或重建配方
  evidence/{benchmarks/, regressions/}    # 可验证证据
  eval/{task_suites/, scorecards/}        # SkillOpt 验证门资产
  policies/{promotion,merge,deprecation}.md
  staging/<member>/<date>/*.json          # Tier 1 入站候选
  tools/{sediment,run_evals,lint,dedup,redact}.py  merge_curator.md
  .github/workflows/ci.yml     # lint + secret-scan + eval-gate + index-build
```

`knowledge/` 把 `global_lessons.md`/ledger 拆成「每条一文件 / 稳定 ID 一行」——不同 ID = 不同文件，git 行级冲突几乎消失，去重/冲突交给 Curator 语义处理（§10）。

### 6.2 skills/ 内部布局

kernel 工作横跨**目录/子模块/文件/函数/招式**等多个维度。若按拓扑建树会组合爆炸，且 rebase 即烂。**唯一原则**：

> skill 树只按「技能种类/稳定性」分层；**比 subsystem 更细的维度（dir/file/function）不是 skill，而是 knowledge**，运行时由 selector 挂载。

每个维度的归宿：

| kernel 维度 | 归宿 |
|---|---|
| 流程 / 跨切面 | `skills/core/<name>/` |
| 优化招式（mechanism）| `skills/technique/<name>/` |
| 子系统（subsystem）| `skills/domain/<subsystem>/` ← **skills 里唯一触及拓扑的层** |
| 目录（dir glob）| domain skill 的 `applies_to.path_globs`（不单独建目录）|
| 文件（file）| `knowledge/targets/<slug>/facts/` + selector |
| 函数（function/symbol）| `knowledge/` idea_ledger + `symbol_selectors` |

只有 3 个维度是真正的 skill 文件夹；file/function 全部沉到 `knowledge/`，从根上消灭爆炸：

```text
skills/
  core/                          # 过程性·拓扑无关·跨所有子系统 (L3, SkillOpt 重点)
    optimization-funnel/  instruction-count-first/  stage-gate-enforcement/
    handoff-contract/  research-discipline/  ab-test-comparison/  ...
  technique/                     # 可复用"优化招式"·拓扑无关·按 mechanism 命名 (非按 target)
    hoist-loop-invariant/  batch-coalescing/  lock-granularity-reduction/
    branch-elimination/  redundant-load-elimination/  inline-tradeoff/  ...
  domain/                        # 唯一触及拓扑·只到 subsystem 粒度
    mm-reclaim/   { SKILL.md, references/ }     # ← 现 memmgr-reclaim 系列
    hyperhold-io/  workqueue-threadpool/  sync-primitives/  sched/  fs/  ...
  _registry/
    skills.yaml                  # 全量索引: name/kind/version/maturity/eval_id/applies_to
    subsystem_selectors.yaml     # subsystem → {path_globs, symbol_selectors} 集中绑定 (单点维护 volatile)
```

每个 skill 是一个文件夹（Anthropic SKILL.md 标准）：`SKILL.md`（简短「何时用+怎么用」，选中即加载）+ `best_skill.md`（SkillOpt 制品）+ `evals/` + `candidates/`（Pareto 前沿）+ `scorecards/` + `references/`（重材料，按需加载）。

**拓扑用 selector 挂载，不写死路径**（domain skill frontmatter）：

```yaml
name: mm-reclaim
kind: domain
applies_to:
  subsystems: [mm/reclaim]
  path_globs: ["mm/vmscan.c", "mm/*reclaim*"]
  symbol_selectors: ["shrink_*", "*_reclaim", "kswapd*"]
requires: [core/optimization-funnel, technique/hoist-loop-invariant, technique/batch-coalescing]
eval_id: eval/task_suites/mm_reclaim_suite
```

`resolver.py`（§12）拿到 target（如 `mm/vmscan.c::shrink_node`）：① selector 匹配 `domain/mm-reclaim`；② 顺 `requires` 拉入 core+technique；③ 检索该函数的 `knowledge/` 挂上。selector 对照**当前 clangd/scip 索引**解析，rebase 后自动重解析、skill 本体不动。

**组合而非枚举**（Voyager 式）：不为每个 `(子系统×招式)` 预制大技能，由 pipeline preset 在加载期组合小技能。`(subsystem × technique)` 矩阵由**加载期组合**消化，不在树里枚举。

**防爆炸：skill vs knowledge 判定三铁律**（三条全过才是 skill，否则归 knowledge）：

1. **可复用**：跨 ≥2 target？只对一个 file/function 成立 → knowledge。
2. **有 eval 信号**：改它能在 eval 上测出行为变化？只是一条事实/裁决 → knowledge。
3. **稳定**：rebase 后不变（招式/流程）？随文件/符号移动 → knowledge。

→ **永不建 per-file / per-function 的 skill。**

**各层 eval**：core 用跨子系统大 suite（最广信号、最难，§15 长杆）；technique 用该招式适用的任务子集；domain 用该 subsystem 代表性任务。三者 `evals/` 独立、互不污染。

### 6.3 消费端 `.opencode/`（共享 + 本地叠加）

```text
.opencode/                # (在业务仓 hm-kernel-llm-opt 内)
  hub/                    # git submodule, pin 到 hm-skill-hub 某版本 (只读)
  local/                  # 本成员执行面产物 (run-evidence + Tier 1 在途记忆)
    runs/<run_id>/{plans,reviews,bench,patches, <target>_design.md}  # 证据, 建议提交留存
    memory/               # 在途工作记忆 (Tier 1): targets/ human_decisions/ idea_ledger/
    sediment_staging/     # hmopt sediment 产出的候选包 (→ PR 到 hub)
  state/current_task.json # 纯运行态, gitignore
  skill-memory.lock       # 锁定 hub 版本 (semver + SHA)
  resolver.py             # 加载时: 先 hub(共享) 再叠加 local(个人在途)
```

这正是 memU `where` 作用域（team vs personal）与 Anthropic project-scope vs personal-scope。

### 6.4 目录迁移归宿表 + 两仓视图

原扁平 `.opencode/` 拆成两半，**无目录原样留老地方**，每个按 §4.3 轴指派：

| 原目录 | 原型 | 新位置 |
|---|---|---|
| `skills/` `agents/` `commands/` `pipelines/` `docs/`(harness 规范) | procedural | **hub** 对应目录 |
| `*_template.md` | procedural（模板是程序）| **hub**（实例落 local）|
| `memory/idea_ledger/` | knowledge | 权威版 **hub** `knowledge/targets/<slug>/`；在途副本 `local/memory/` |
| `memory/targets|subsystems|global_lessons` | knowledge | 蒸馏→ **hub** `knowledge/`；在途副本 `local/memory/` |
| `memory/human_decisions/` | run-evidence→knowledge | 原始时间线留 `local/`（需脱敏）；稳定摘要→hub `decisions/` |
| `state/bad_plans.md` | knowledge | 蒸馏→ **hub** `knowledge/global/anti_patterns/` |
| `state/current_task.json` | run-evidence（纯态）| **仅本地** gitignore |
| `plans/` `reviews/` `bench/` | run-evidence | **业务仓** `local/runs/`；validated delta/反模式→hub |
| `patches/` | run-evidence | **业务仓**（随代码），不入 hub |

```
┌─ hm-skill-hub  (资产面 · 共享 · semver) ─────────────────────────────────┐
│   PROCEDURAL (引擎 B)                       KNOWLEDGE (引擎 A)            │
│     skills/{core,technique,domain}  agents/   knowledge/   evidence/      │
│     commands/  pipelines/  docs/              ▲ 只放"蒸馏精华"            │
└──────────▲─────────────────────────────────────┼────────────────────────┘
           │ (1) pin: submodule + lock (只读)      │ (3) promote: 蒸馏 + eval 门
┌─ 业务仓 /.opencode/ (执行面) ────────────────────┴───────────────────────┐
│   hub/  ← submodule 只读 pinned                                           │
│   local/  runs/<id>/{plans,reviews,bench,patches}  memory/  sediment_staging/ │
│   state/current_task.json (gitignore)   skill-memory.lock   resolver.py   │
└────────────────────────────────────────────────────────────────────────────┘
```

**路径兼容（Phase 0 必处理）**：现 harness 硬编码 `.opencode/skills/X.md` 等相对路径，移入 `hub/` 会破坏。二选一：① symlink `.opencode/skills→hub/skills` 保旧路径（最小改动）；② 批量改写 + `resolver.py` 统一解析。推荐先 ① 兜底再迁 ②。

---

## 7. 数据模型与 Schema

**类型化记忆**——每条 Knowledge 记录声明 `type`，五类之一：

| type | 含义 |
|---|---|
| `fact` | 稳定结构事实 |
| `rule` | 操作规则 |
| `pattern` | 可复用正向模式 |
| `anti_pattern` | 反模式 |
| `playbook_step` | 流程步骤片段 |

**`memory_item` 关键字段**（完整 JSON-Schema 在 `schemas/`）：

```
id(稳定,前缀 F/G/A/R) · type · title · body
scope{level: function|call-site|data-flow|subsystem|architectural|global, subsystem, target_slug}
source[]{kind: commit|review|bench|doc|run_id, ref}   ← 必填, 无出处即拒
evidence{delta_pct, compare_level, confirmations}
maturity(L0-L3) · status(active|superseded|deprecated) · score
invalidation(失效条件, 如 "rebase 后须重校 offset")
supersedes[] · valid_from · valid_until · contributor · created_at
```

**skill frontmatter**（兼容 SKILL.md，详见 §6.2）：`name/kind/version/maturity/applies_to/requires/eval_id/owners/status`。

**技能更新清单**（每个 skill PR 必附，否则 CI 拒）：绑定 `edit_ops` + `task_suite` + `metrics{pass_rate, instr_count_delta, regression_rate}` + `baseline_version`。

**idea_ledger**：沿用现有结构（稳定 ID、状态机、永不删除），仅 ① 字段 JSON 化便于机器合并；② 外置到 hub；③ 由 Curator 跨成员合并。

---

## 8. 沉淀时机与晋升

**Tier 0→1（蒸馏）触发**：复用已有收口点——pipeline decision 阶段、人机会话 "done"、每个 auto-iterate pass 末。产物落 `local/sediment_staging/*.json`（标 `maturity: L1`）。

**Tier 1→2（晋升）触发**（满足其一 + 过 §9 三门）：① ≥2 个独立任务复现收益；② 单任务收益显著且有 bench 证据；③ 高复用失败教训（→ anti_pattern）。

**贡献节奏**：自动暂存（持续）+ 批量 PR（每周/里程碑），`hmopt sediment` 打包候选成单个「沉淀 PR」，便于统一去重、避免刷屏。

---

## 9. 质量门（三道）

```
候选(L1) → [门1 Schema/Lint/脱敏] → [门2 证据] → [门3 策展+eval] → 稳定(L2/L3)
            CI 自动                    自动        Curator + 人 + eval-gate
```

1. **Schema/Lint/脱敏**：过 §7 schema；`redact.py` + CI secret-scan 强扫设备序列号/key，命中即拒（团队仓泄密是放大事故）。
2. **证据**：知识需引用（`validation_path`/`delta_pct`/`confirmations≥N`）；技能编辑需 eval 结果。无证据 → 留 L1。
3. **策展 + eval**：Curator 预处理去重/冲突/泛化（§10），再由**双评审人**签字（1 领域 + 1 流程）。技能额外过 eval-gate（留出 suite 严格变好）。**不设豁免**：破例只能降级为 L1 候选 + owner 签字 + 复核。

**打分**（晋升/检索排序 + 衰减）：`score = w1·证据强度 + w2·确认数 + w3·新近度 + w4·泛化范围 − w5·反例 − w6·陈旧度(invalidation 触发后衰减)`。

---

## 10. 合并机制（两套引擎）

### 10.1 引擎 A — Knowledge：集合并 + 去重 + 冲突消解（绝不行级合并）

```
for item in incoming:
    dup = near_duplicate(item, hub)        # embedding 相似度
    if dup: merge_provenance(dup, item); confirmations += 1; continue
    conflict = contradiction(item, hub)    # 同 (target, mechanism) 断言相反
    if conflict:
        if stronger_evidence(item):        # 证据/新近度加权 (Zep 双时态)
            conflict.status = "superseded"; item.supersedes = [conflict.id]; add(item)
        elif high_risk: escalate_to_human()
        else: drop_with_citation(item)
    else: add(item)
```

**CRDT 纪律**：追加 + tombstone（`active/superseded/deprecated`）而非删除；双时态 `valid_from/until` 接住「rebase 使 offset 失效」。

### 10.2 引擎 B — Skills：SkillOpt 验证门 + GEPA Pareto（绝不集合并）

```
def merge_skill_edit(skill, edit):
    if edit in bad_edits: return REJECT          # 被拒编辑缓冲
    edit = clip_to_budget(edit, textual_lr)      # 文本学习率: 有界编辑
    cand = apply(skill, edit); s = run_evals(cand, suite)
    if s.strictly_better_than(skill.score):      # 严格变好才接受
        skill = cand; write_scorecard(s)
    else: bad_edits.append(edit)
    pareto = update_pareto(pareto, cand, per_instance_scores)   # 保留互补候选
    return skill, pareto
```

**为何 Pareto**：N 个成员各提编辑时，单一全局 eval 分会让「互补但互斥」的编辑塌缩到局部最优。Pareto 前沿保留「各自在某些实例上最优」的候选（`candidates/`），定期合并互补 lesson——**这才是「人人沉淀、统一汇入而不互相覆盖」的正解**。文本学习率 = 每次发布的有界编辑预算；慢更新 = 按发布周期批量合并。

> **一句话**：知识靠「集合并+去重+冲突消解」；技能靠「验证门竞争式编辑 + Pareto」。两类资产，两台引擎。

---

## 11. 闭环优化作业（定时）

nightly/weekly「Skill/Memory 优化作业」：

```
(1) Collect    聚合各项目候选 → staging/
(2) Normalize  按 schema 标准化 + 去噪 + 脱敏
(3) Cluster    embedding 聚类 (引擎 A 去重前置)
(4) Optimize   对 skills 跑 SkillOpt 有界编辑 (引擎 B)；早期半自动: 自动提 PR + 人工合并
(5) Validate   留出 suite A/B, 出 scorecard
(6) Promote    仅升收益版本, semver + tag, 更 registry.yaml
(7) Broadcast  生成 release notes, 供业务仓 pin
```

**早期安全约束**：第 (4) 步必须接 `bad_edits` + Pareto + 脱敏，**默认半自动**（自动提 PR、人工合并），积累信任后再放开。

---

## 12. 消费与集成

- **版本锁定**：`skill-memory.lock`（semver + SHA）固定 hub 版本，等价 package lockfile，可复现、防漂移；每次 run 记录消费版本。
- **运行时叠加**：`resolver.py` 先读 hub（共享）再叠加 local（个人在途），互不污染（memU `where`）。
- **故障降级（可用性）**：hub 以 submodule pin 在本地（vendored 副本），**中央仓宕机不阻塞**；resolver 检测不可达即回退上次成功快照并告警。
- **跨工具**：SKILL.md 开放标准，可同时被 OpenCode / Claude Code / Codex 消费，hub 即「团队私有 skill marketplace」。

---

## 13. 治理 · 稳定 · 可用（汇总）

`policies/` 三份一等文档固化规则：`promotion`（§8 触发 + §9 三门 + 晋升路径）、`merge`（§10 两引擎 + 双评审 + 无豁免 + CODEOWNERS）、`deprecation`（失效治理）。发布节奏：每周小版本、每月稳定版，`skills/core/` 走更严评审。

| 保障 | 机制 |
|---|---|
| 可回滚 | 每次更新有 tag + scorecard；`git revert` 即回退 |
| 抗回归 | **CI eval-gate**：不达标禁止发布（反喂安全的根本机制）|
| 抗破坏改写 | 文本学习率 + 被拒编辑缓冲 + 慢更新（SkillOpt 三件套）|
| 可审计 | 每条回链来源 + reviewer + scorecard |
| 抗污染 | 候选层（L1）与稳定层（L2/L3）物理隔离 |
| 多版本共存 | 不同项目 pin 不同版本，逐步升级 |
| 高可用 | lockfile + 本地兜底 + resolver 降级 |
| 失效治理 | `superseded/deprecated` 状态 + 双时态 + 定期清理（保留可审计，不物删）|
| 防泄密 | 脱敏门 + CI secret-scan |

---

## 14. 分阶段路线图

| 阶段 | 周期 | 目标 | 交付物 | 风险 |
|---|---|---|---|---|
| **0 抽取** | 1–2w | 零行为变更跑通双仓 | 切 `skills/agents/pipelines/commands/docs` 到 hub + submodule pin；**路径兼容**（symlink/改写，§6.4）；仓骨架 + schemas + registry | 低 |
| **1 蒸馏** | 2–3w | Tier0→1 结构化 | `hmopt sediment`；`memory export` 转标准对象 | 低 |
| **2 策展+合并** | 3–6w | 知识合并上线（引擎 A）| Curator + lint/secret-scan/dedup CI；`policies/` 三文档 | 中 |
| **3 eval 门** ★ | 6–10w | 安全反喂（引擎 B）| **建 core task suite**（长杆）+ CI eval-gate + scorecard；半自动优化器 | **高** |
| **4 自动优化** | 10w+ | 闭环自动迭代 | 定时作业（§11）；发布节奏；`skill-memory.lock` 防漂移 | 中 |

---

## 15. 风险与缓解

| 风险 | 缓解 |
|---|---|
| **eval 套件是长杆**（真机 A/B 慢/贵/噪声）| Phase 3 重点；先静态代理指标 + 小样本真机起步，逐步加密；诚实标注为关键路径 |
| 过度/不足沉淀 | 打分 + 衰减 + N 次确认阈值；L0–L3 过滤 |
| 密钥泄露 | 脱敏门 + CI secret-scan（强制）|
| 热点文件合并争用 | 每记录一文件 + 稳定 ID，规避行级合并 |
| 两类资产混用一引擎 | §4.1 强制分治 |
| 多成员编辑塌缩 | GEPA Pareto（§10.2）|
| skills 维度爆炸 | core/technique/domain 三层 + file/function 归 knowledge + selector（§6.2）|
| eval 豁免漏洞 | 无豁免；破例只能降级候选 + 签字 + 复核 |
| 路径硬编码迁移 | Phase 0 symlink 或改写 + resolver（§6.4）|

---

## 16. 附录

**关键 CLI（拟新增到 `hmopt`）**

```bash
hmopt sediment [--bundle --open-pr]    # Tier0→1 蒸馏 / 打包候选提 PR
hmopt skill-lock --update <semver>     # 更新 skill-memory.lock
hmopt skill-eval <skill> --suite <s>   # 本地跑技能 eval, 出 scorecard
```

**术语表**

| 术语 | 含义 |
|---|---|
| Tier 0/1/2 | 运行轨迹 / 候选沉淀 / 核心共享 |
| L0–L3 | draft / candidate / stable / core |
| 引擎 A / B | 知识合并 / 技能合并 |
| 三原型 | procedural-shared / knowledge-curated / run-evidence-local |
| 文本学习率 | 每次发布的有界编辑预算（SkillOpt）|
| Pareto 前沿 | 各自在某些 eval 实例上最优的一组候选（GEPA）|
| eval-gate | 发布前留出验证回归门——反喂安全的根本机制 |
| selector | skill frontmatter 里对照代码索引解析的拓扑绑定（subsystem/glob/symbol）|

---

## 17. 待团队拍板

1. hub 仓命名与归属、放哪个 GitHub org。
2. eval ground truth：纯真机 A/B vs 静态代理 vs 混合（决定 Phase 3 工期）。
3. 是否采用 `technique/` 层（现招式隐含在 funnel scope 标签里）。
4. 检索后端：faiss 本地文件 vs pgvector（对齐现有 `storage/`）。
5. `skills/core/` owner 团队与晋升评审人。
