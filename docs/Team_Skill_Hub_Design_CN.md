# 团队级 Skill / Memory 仓库闭环方案设计（Team Skill Hub）

| 项 | 值 |
|---|---|
| 文档状态 | Draft v2.3（待团队评审） |
| 日期 | 2026-06-09 |
| 适用范围 | `.opencode/` harness 的团队化演进；新增独立中央仓 `hm-skill-hub` |
| 语言策略 | 散文 zh-CN；路径 / 字段名 / 代码 / CLI / commit 一律英文 |
| 关联文档 | `.opencode/docs/harness_engineer_system.md`、`.opencode/docs/memory_system.md` |
| 图示 | `docs/Team_Skill_Hub_Design_Diagrams_CN.md`（7 张 Mermaid：闭环 / 双引擎 / 沉淀漏斗 / skills 布局 / 运行时组合 / 路线图 / 读路径）|
| 修订 | v2.3（评审反馈，2026-06-09）：① §6.1 收敛「每条一文件 + frontmatter + 路径编码 scope + CI 一致性校验」，消除与 Phase 0 多记录示例的矛盾；② §7 加 `subsumes[]/subsumed_by[]` 字段 + frontmatter 必带全字段约束；③ §10.1 引入统一「合并关系分类表」（dup / contradiction / temporal / conditional / subsumption / selector / evidence 七路），10.1.a 禁止对时态/条件/selector 冲突 delete，10.1.b 加 subsumption（LLM 蕴含判定）；④ §11.5 打通 subsumption → 晋升 + ≥2 实例防伪泛化；⑤ §17 议题 7 升为 P1 前置阻塞 + 加路径编码 scope 决策。v2.2（mem0/EverOS 调研）：§3 调研行 + §8 LLM 显著性 pass + §10.1 两级合并 + §11.5 晋升检测器 + §12 检索与运行时组合改写 + §14/§15 同步。v2.1：§6.2 skill/knowledge 判定。v2.0：§6.2 skills 布局 + 精简 |

---

## 0. TL;DR

现有 `.opencode/` 把**两类性质不同的资产**混在一起：

- **Skills（过程性指令）**——可被 eval 衡量、可被优化的「程序」。
- **Knowledge（事实/记忆）**——不断追加、需去重与冲突消解的「学习到的状态」。

**方案脊柱**：把两类拆开、用两套合并引擎分治，再通过一条带验证门的「沉淀漏斗」把每个成员的本地经验汇入一个独立、semver 版本化的中央仓 `hm-skill-hub`，并以 pinned 版本反向喂回 pipeline，形成闭环。

- **Skills** → 用 **SkillOpt** 治理：技能文档 = 冻结模型的「可训练外部参数」，改动必须在留出 eval 套件上**严格变好**才接受；用 **GEPA Pareto** 解决多成员编辑塌缩。
- **Knowledge** → 用 **memU/Mem0/Zep** 治理：分层、类型化、稳定 ID、**追加+去重+冲突消解**（非 git 行级合并）、双时态保留被取代记录。

四个保证：**可迭代**（定时优化作业）、**可沉淀**（三层 + L0–L3）、**可闭环**（消费→蒸馏→晋升→发布→再消费）、**可复用 + 稳定可用**（SKILL.md 标准 + semver + lockfile + eval-gate + 本地兜底 + 脱敏门）。

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
| **Mem0 / Mem0g**（v0.1.x 论文版；v3 OSS 已退化，见下）| 两阶段抽取→冲突检测+消解：Phase1 用 `FACT_RETRIEVAL_PROMPT` 抽事实，Phase2 对候选取最近 k 条做 ADD / UPDATE / DELETE / NOOP 工具调用消解 | **本地在线**消解（引擎 A 第一级，§10.1.a）+ Curator 合并器灵感 |
| **EverOS**（EverMind-AI，v1.0 / 2026-06）| markdown 为真相源；六类记忆（含一等公民 procedural Skill）；同步轻量抽取 + 异步 OME 离线重整理；技能/画像 clustering 触发器；cascade 增量重嵌；LanceDB 一次查询并跑 BM25+向量+scalar 过滤；明确**不含**冲突/衰减/质量门 | **markdown 为真相源 + 索引可重建**模式；**混合检索**模板（§12）；**晋升候选 clustering**（§11.5）；反面对照：缺治理层正是 hub 的差异化 |
| **Zep / Graphiti** | 双时态，事实带失效时间 | 被取代记录不删除（`superseded`）|
| **ExpeL** | 成败池 → 抽取洞见 ADD/UPVOTE/DOWNVOTE/EDIT | 晋升打分与衰减 |
| **GEPA** | 反思式进化 + Pareto 前沿（保留互补候选，非单一最优）| 避免多成员编辑塌缩 |
| **Voyager** | 增长式技能库，入库前自验证，可组合 | 技能库范式 + 组合复用 |
| **Anthropic Agent Skills** | SKILL.md 开放标准；plugin 分发；project-scope 版本控制共享；marketplace 安全扫描 | 分发与互操作标准 + 治理 |

**⚠️ mem0 v3 OSS 提示**（2026-04 起）：开源 `main.py` 将两阶段塌缩为「`ADDITIVE_EXTRACTION_PROMPT` + content-hash 去重」的 **ADD-only** 单遍流；LLM 驱动的 UPDATE/DELETE 与图记忆**仅保留在付费 Platform**。规划层面意味着：本设计在 §10.1.a 引用 mem0 时，OSS 版只能拿到「抽取 + 哈希去重 + 索引/检索」，**智能 UPDATE/DELETE/NOOP 消解逻辑需要按 v0.1.x 论文 prompt 自己复刻**（或评估付费 Platform）。

**结论**：SkillOpt 给「技能优化工程化范式」，memU 给「记忆资产组织范式」，mem0/EverOS 给「在线抽取+混合检索」工程模式，**而它们都缺乏跨成员策展闭环（semver/eval-gate/L0–L3/双评审）**——这正是本设计的差异化护城河。最契合做法：保留 `.opencode/` 为执行面，新增 hub 为资产面；**本地层**借鉴 mem0/EverOS 拿到「便宜在线消解 + 混合检索」的延迟与成本红利，**中央层**保留 hub 自研重型治理（CI + eval-gate + 双评审）把「经验沉淀」变成「可验证迭代」。

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

> **v2.2 补**：引擎 A（Knowledge）在工程上分两级——**本地在线**（mem0/EverOS 式：每次收口就跑 ADD/UPDATE/DELETE/NOOP，保持 `local/memory/` 不烂）+ **中央批量**（Curator + CI + eval-gate）。详见 §10.1。引擎 B（Skills）始终是中央批量，**不**在本地做（个人 eval 噪声大，反易污染）。

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
| **knowledge-curated** | facts / rules / patterns / anti_patterns / idea_ledger（稳定招式可毕业为 technique skill，§6.2）| **hub** `knowledge/`（只放蒸馏精华）|
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

**存储形态收敛（v2.3，对应 §17 议题 7 / Phase 0.5 阻塞项）**：`knowledge/` 一律「**每条记录一文件**」，文件 = YAML frontmatter（全 schema 字段）+ markdown body。不同 ID = 不同文件，git 行级冲突几乎消失，去重/冲突/subsumption 交给 Curator 语义处理（§10）。

> ⚠️ **收敛动作**：Phase 0 示例（`A001-*.md` 是 `### A001` 多记录堆在一个 category 文件、字段自定义为 `lesson/applies_when/...`）与本原则**不一致**，且与 `memory_item.schema.json` 字段不对齐。Phase 0.5 必须**先**收敛（§17 议题 7，列为 P1 前置阻塞）：① 一记录一文件；② frontmatter 用标准 schema 字段（不允许每类 markdown 自扩字段）；③ `parse_memory.py` 输出标准 schema object。

**路径即 scope（v2.3）**：文件路径**编码** scope，与 frontmatter 的 `scope` 字段**冗余且必须一致**，CI 强校验（不一致即拒）。这让 scalar 过滤（§12.1）可先按目录粗筛、再读 frontmatter 精筛：

```text
knowledge/
  global/{heuristics,anti_patterns,bad_plans,validation_pitfalls}/<ID>.md   # scope.level=global
  subsystems/<subsystem>/<ID>.md                                            # scope.level=subsystem
  targets/<slug>/{facts,decisions}/<ID>.md                                  # scope.level=function|call-site|...
  targets/<slug>/idea_ledger/<Lxxx>.md                                      # 每条 idea 一文件
  index/                       # 派生缓存：向量 / BM25 / scalar 清单 + 重建配方（manifest.yaml）
```

> **真相源不变式（EverOS 印证）**：`*.md` 与其 YAML frontmatter 是 hub 的**唯一真相源**，`index/` 下的向量/BM25/scalar 全部是**派生缓存**——删掉 `index/` 不丢任何知识，从 markdown 树整树重建即可。这条不变式同时给出三件保障：① 灾难恢复路径自然；② 索引后端可替换（faiss / pgvector / LanceDB 任意切换不影响数据）；③ 任何成员可手工 git 编辑、PR review 纯文本可读。**禁止**任何"先入索引、markdown 滞后"的写路径。

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

**防爆炸：skill vs knowledge 判定**

**首要判据（先问这条）**：**做法/流程**（AI 照着*执行*，靠*改写措辞*调优、eval 衡量好坏）→ **skill**；**事实/结论/教训**（AI *查阅*，靠*增删纠错条目*维护，不靠改措辞）→ **knowledge**。一刀切：「查清单」是 skill，「清单内容」是 knowledge。

下面三条只是**排除项**（任一不满足 → 必是 knowledge；三条全过也不直接等于 skill，仍以首要判据为准）：

1. **可复用**：只对一个 file/function 成立 → knowledge。
2. **稳定**：随文件/符号移动（rebase 即变）→ knowledge。
3. **可优化**：措辞不是你会反复调、eval 测不出差异 → knowledge。

例：`bad_plans` / `global_lessons` 三条都过、却仍是 **knowledge**（被查阅的教训；"出方案前先查 bad_plans 去重"这条*流程*写在 `optimization-funnel` 技能里）。某条教训若稳定成"固定步骤 + 可复用 + eval 可测"的招式，可**毕业**升入 `technique/`（knowledge 是原矿，technique 是提纯成品）。

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

> `pattern` / `playbook_step` 是被*查阅*的记录条目（"有这么个招式/步骤"）；当它稳定成可*执行*、eval 可测的流程，毕业为 `technique` / `core` 技能（§6.2 首要判据）。

**`memory_item` 关键字段**（完整 JSON-Schema 在 `schemas/`）：

```
id(稳定,前缀 F/G/A/R) · type · title · body
scope{level: function|call-site|data-flow|subsystem|architectural|global, subsystem, target_slug}
applies_when(条件适用范围, 用于「条件分歧」共存判定, §10.1)   ← v2.3
source[]{kind: commit|review|bench|doc|run_id, ref}   ← 必填, 无出处即拒
evidence{delta_pct, compare_level, confirmations}
maturity(L0-L3) · status(active|superseded|deprecated) · score
invalidation(失效条件, 如 "rebase 后须重校 offset")
supersedes[] · superseded_by[]                         ← 时态/矛盾关系 (双时态)
subsumes[] · subsumed_by[]                             ← v2.3 泛化包含关系 (§10.1 / §11.5)
valid_from · valid_until · contributor · created_at
```

> **frontmatter 约束（v2.3，CI 强校验）**：每条 knowledge 落盘文件的 YAML frontmatter **必须**含上述全部 required 字段，**不允许**每类 markdown 自扩字段；**文件路径编码的 scope 必须与 frontmatter `scope` 一致**（§6.1）。`subsumes[]/subsumed_by[]` 与 `supersedes[]/superseded_by[]` 同为**关系边**——这是未来若引入图层（mem0g / Graphiti）的第一批落地边，现阶段仅作字段、不建图存储。

**skill frontmatter**（兼容 SKILL.md，详见 §6.2）：`name/kind/version/maturity/applies_to/requires/eval_id/owners/status`。

**技能更新清单**（每个 skill PR 必附，否则 CI 拒）：绑定 `edit_ops` + `task_suite` + `metrics{pass_rate, instr_count_delta, regression_rate}` + `baseline_version`。

**idea_ledger**：沿用现有结构（稳定 ID、状态机、永不删除），仅 ① 字段 JSON 化便于机器合并；② 外置到 hub；③ 由 Curator 跨成员合并。

---

## 8. 沉淀时机与晋升

**Tier 0→1（蒸馏）触发**：复用已有收口点——pipeline decision 阶段、人机会话 "done"、每个 auto-iterate pass 末。产物落 `local/sediment_staging/*.json`（标 `maturity: L1`）。

**两段抽取（v2.2）**：

1. **规则映射抽取器**（确定性、必跑）：`extractors.py` 把 bench delta → fact、review 否决 → anti_pattern、ledger 状态机变更 → idea record。保证 `delta_pct / compare_level / source[]` 等**结构化字段不丢**。
2. **LLM 显著性 pass**（启发式、可关）：取规则抽取剩余的自由文本（design 摘要、reviewer 笔记、人机决策对话），跑一次 `FACT_RETRIEVAL_PROMPT` 风格的抽取，捕捉**不落规则模板**的可复用洞见（mem0 / EverOS 都用这条；只用 LLM 抽容易漏结构化指标，只用规则抽容易漏"这条其实可复用"的非典型洞见，两段叠加最稳）。LLM pass 产物默认 `confidence: tentative`，需后续 confirmations 才升 maturity。

**收口节奏（EverOS 印证）**：同步收口点只做**便宜抽取**（≤ 100ms 量级，规则 + 选填轻量 LLM），重的去重 / 显著性聚合 / 跨 run 关联交给**异步离线作业**（§11）。同步流程**不阻塞**主 pipeline。

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

v2.2 起拆成**本地在线** + **中央批量**两级。同一个分类器核心，跑在两个不同时机和不同权限层级上。

**为什么两级**：只有中央 Curator 会让 `local/memory/` 在到达批处理之前一周积重难返——重复、自相矛盾、检索质量劣化，到 Curator 时再去清理已经晚了。mem0 / EverOS 印证：在线消解可便宜跑（小 candidate set、近似最近邻），是检索质量的前置条件。

#### 10.1.0 合并关系分类表（v2.3 核心）

合并决策**不是**「重复? / 矛盾?」二分，而是把 incoming 与最近 k 条已有记录的**关系**分到下面七类之一。**铁律：除「明确矛盾且新证据更强」外，任何分支都不物理 delete**——这是评审反馈「别把历史事实误删」的根本保障。

| 关系 | 判定 | 处理 | 谁来判 | **绝不** |
|---|---|---|---|---|
| **duplicate** | 语义近重复、同 scope 同结论 | 合并 source[]，`confirmations += 1` | hash + embedding（廉价）| — |
| **contradiction** | 同 (target, mechanism, **同条件**) 断言相反 | 新证据强 → 旧记 `superseded` + `valid_until`，`superseded_by` 互链；否则 escalate | embedding + LLM | 删旧记 |
| **temporal staleness** | 旧记**曾对、现已过时**（如新 kernel 版本行为变了）| 旧记 `superseded` + `valid_until=now`，**保留可审计** | LLM（看 valid_from / 版本）| **当作错误 delete** |
| **conditional divergence** | 两条**都对、适用条件不同** | **共存**，各自写明 `applies_when` / `scope` | LLM | 当作矛盾去重 |
| **subsumption（泛化包含）** | 一条是另一条的**泛化**（B 概括 A）| **都留**：A 作 target-level evidence；B 升 pattern/technique 候选；A 进 B 的 `source[]` + 互链 `subsumes/subsumed_by` | **LLM 蕴含判定** | 把 A 去重吞进 B |
| **selector drift** | 同 symbol **rebase 后路径/偏移变了** | **重解析 selector**、更新 `invalidation`，知识本体不动 | clangd/scip 索引 | 删该知识 |
| **evidence divergence** | 同 mechanism 同 delta、**`compare_level` 不同** | **合并**，按 `compare_level` 消歧（total/process/function 不可直接比）| 规则 | 当作矛盾 |
| **novel** | 与最近 k 条无上述关系 | ADD | — | — |

**分层归属**：`duplicate / temporal / conditional / contradiction / evidence` 廉价或单 LLM call → **两级都跑**（本地 10.1.a + 中央 10.1.b）；`selector drift` 依赖代码索引 → 本地 resolver 加载期 + 中央 CI；**`subsumption` 需 LLM 蕴含判定、较贵 → 仅中央 10.1.b**（本地延迟预算扛不住，且泛化是跨成员信号）。

#### 10.1.a 本地在线（每个收口点跑一次，每成员独立）

```
# 触发：sediment 收口点。延迟预算: 单条 ≤ 1 LLM call + 1 ANN query (~1-3s)
for item in just_sedimented(local):
    if hash_seen(item): merge_provenance(...); continue   # cheap dedup, 先于 LLM
    nearest = vector_search(local.index, item, k=5, filter=scope)
    rel = classify_relation(item, nearest)      # §10.1.0 表（本地只跑廉价 5 类）
    apply(rel, local.memory)                     # 见下「禁止 delete」纪律
```

- **作用域**：仅在 `local/memory/<member>/` 内消解，**不**跨成员；**不跑 subsumption**（留中央）。
- **关键纪律（评审反馈 ③）**：本地 `apply` 对 `temporal / conditional / selector / evidence` 四类**一律不 delete**——temporal → `superseded`+`valid_until`；conditional → 共存（写 `applies_when`）；selector → 重解析 + 更 `invalidation`；evidence → 按 `compare_level` 合并。唯一会写 tombstone 的是「contradiction 且新证据更强」，且 tombstone 是 `superseded` 不是物删。**本地 false-delete 必须 ≈ 0**（P1-8 PoC 硬指标）。
- **依赖**：可用 `pip install mem0ai` 拿到「索引 + ANN + 去重」基础设施；**关系分类 prompt 自带**（参照 mem0 v0.1.x 论文版 `DEFAULT_UPDATE_MEMORY_PROMPT` 扩展到七路，因为 mem0 v3 OSS 已经只剩 ADD-only）。
- **markdown 仍是真相源**：在线消解的产物**直接写回 markdown frontmatter + body**，索引由 cascade 增量重建（§12）。

#### 10.1.b 中央批量（PR / nightly，跨成员）

```
# 触发：沉淀 PR 或 nightly Curator。延迟预算: 分钟级
for item in incoming_from_all_members:
    rel = classify_relation(item, hub)     # §10.1.0 全七路，含 LLM 蕴含判定
    match rel.kind:
        case duplicate:    merge_provenance(rel.target, item); confirmations += 1
        case temporal:     rel.target.status="superseded"; rel.target.valid_until=now(); link(item, rel.target)
        case conditional:  add(item)        # 共存，校验两者 applies_when 不重叠
        case evidence:     merge_by_compare_level(rel.target, item)
        case contradiction:
            if stronger_evidence(item): rel.target.status="superseded"; item.supersedes=[rel.target.id]; add(item)
            elif high_risk: escalate_to_human()
            else: drop_with_citation(item)
        case subsumption:  # ← v2.3 新增第三类（评审反馈 ④）
            general, specific = orient(item, rel.target)   # 谁泛化谁
            specific.subsumed_by += [general.id]; general.subsumes += [specific.id]
            general.source += specific.as_source()         # specific 成为 general 的证据，不被吞
            emit_promotion_signal(general)                 # → §11.5（≥2 实例才晋升）
        case novel:        add(item)
```

- **作用域**：跨所有成员、跨 hub 全量 knowledge。阈值比本地严：相似度收紧、冲突走双评审（§9 门 3）。
- **subsumption ≠ duplicate ≠ contradiction**（评审反馈 ④）：例「`shrink_node` 中 hoist `sc->priority` 降低重复读取」(A，target-level) vs 「mm reclaim 热循环中 loop-invariant 状态应 hoist 出循环」(B，pattern-level)——B 是 A 的**泛化**。处理：A 留为 target 证据、B 升 pattern/technique 候选、**A 作 B 的 `source`/evidence 而非被去重吞掉**。这是 `knowledge → technique skill` 毕业通道的引擎。
- **防伪泛化**：subsumption 只立刻**建链**（廉价、安全）；泛化记录 B 真正**晋升**为 technique 仍需 §11.5 的 **≥2 个不同被包含实例**门 + §9 三门。单个 A 不足以喂出一条 technique。
- **新增职责**：**跨成员同语义簇合并**（不同人对同一事实的不同措辞）→ 合并出处 + 累加 confirmations。

**CRDT 纪律（贯穿两级）**：追加 + tombstone（`active/superseded/deprecated`）而非删除；双时态 `valid_from/until` 接住「rebase 使 offset 失效」。**本地** tombstone 不立即 push；**中央** tombstone 是发布 artifact。

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

### 11.5 晋升候选自动检测器（v2.2 新增）

主流程的 L1→L2、knowledge→technique skill 晋升靠人工 PR 启动，**信号容易被埋**。EverOS 的 `trigger_skill_clustering.py` / `trigger_profile_clustering.py` 验证了「聚类**重复模式** → 自动开 PR 交人评审」是个工程可行的中间档：**只自动化候选检测，决策仍由 §9 门人评审**——治理不让步、人不淹没。

**两路输入**（v2.3）：
- **聚类信号**：embedding 聚类 hub knowledge（按 mechanism + scope 维度）。
- **subsumption 信号**（§10.1.b 喂入）：某条泛化记录 B 的 `subsumes[]` 累积到 **≥2 个不同被包含实例**（不同 target / 不同 contributor）——这是比纯聚类更强的毕业信号（已显式建立「具体→泛化」关系）。

```
(1) Gather       收 (a) embedding 簇 + (b) subsumes[] ≥ 2 的泛化记录
(2) Threshold    簇内 confirmations 总和 ≥ N（默认 3）且跨 ≥ 2 contributors；
                 subsumption 路要求 ≥ 2 个不同被包含实例（防伪泛化）
(3) Distill      调 LLM 把簇/泛化记录蒸馏为「招式 + 适用条件 + 证据列表（含被包含实例）」
(4) PR-Open      自动开 promotion PR（标签 promote-candidate），CODEOWNERS 接力
(5) Guard        晋升 PR 仍走 §9 三门（schema/evidence/curation+eval）
```

**两个适用场景**：
- **L1 → L2**：staging 区跨成员命中同一事实 N 次 → 提议晋升到 hub `knowledge/global/` 或 `knowledge/subsystems/`。
- **knowledge → technique skill**：同一 mechanism 下的 anti_pattern/heuristic 簇 ≥ N，**或**一条 pattern 已 `subsumes` ≥2 个 target-level 实例 → 提议**毕业**为 `skills/technique/<mechanism>/`（§6.2 首要判据「做法/流程」具备时）。被包含的具体实例**保留**为该 technique 的 `evidence`，不删。

**纪律**：检测器只能**提建议**、**不能**自己合并；任何 promote-candidate PR 都必须由人显式 approve，无豁免。

---

## 12. 检索与运行时组合（v2.2 改写）

**v2.1 之前**这一节只写「先 hub 再 local + RAG」一句话；v2.2 把整个**读路径**补完整——这是 mem0 / EverOS 的全部价值区，原稿严重欠设计。

### 12.1 三类检索 query，一条混合检索栈

`resolver.py` 在 pipeline 各阶段对 hub + local 发起检索。**输入分三类**，**底层栈同一套**：

| query 类型 | 触发点 | 输入 | 主要消费者 |
|---|---|---|---|
| **target-anchored** | research / plan / code 阶段 | 当前 target slug + symbol（如 `mm/vmscan.c::shrink_node`）| domain skill selector 命中后挂载 knowledge |
| **mechanism-anchored** | plan-review / code-review | 候选 mechanism（`hoist-loop-invariant` 等）| technique skill 上下文 + 相关 anti_pattern |
| **free-form** | 任意时刻，agent 显式提问 | 自由文本 | 兜底通用检索 |

**混合检索栈（EverOS LanceDB 模式）**：

```
def retrieve(query, scope_filter, k=5):
    # 1) Scalar 预过滤（schema 字段直接命中，廉价）
    cands = scalar_filter(
        index, status="active",
        maturity_in={"L2","L3"},  # 默认排除 L0/L1, 灰度可调
        scope=scope_filter,        # subsystem / target_slug / level
    )
    # 2) Hybrid score: BM25 + 向量 cosine + 实体匹配 + 时序新近度
    v_scores  = vector_topk(cands, embed(query), k=4*k)
    bm_scores = bm25_topk(cands, query, k=4*k)
    ent_bonus = entity_match_bonus(cands, extract_entities(query))
    fused     = rrf_fuse(v_scores, bm_scores) + ent_bonus
    # 3) score 字段加权（§9：晋升打分喂回排序，新近 / confirmations 高的优先）
    fused    *= sigmoid(item.score)
    return topk(fused, k)
```

四件事是 mem0 / EverOS 教给我们的：① **scalar 过滤先于向量**（成本量级差异巨大，schema 现有的 `scope.level / maturity / status / scope.subsystem` 直接拿来用）；② **BM25 + 向量融合**（纯向量在术语命中场景拉胯——`shrink_node` 这种符号名向量近似很差，BM25 救场）；③ **`score` 字段（§9）必须喂回排序**，目前只用于晋升排序、读路径没接上，是个明显 bug；④ **每阶段有 token 预算**，不是检索越多越好——mem0 论文给出 7K vs 25K tokens/query 的对比，过量上下文会反向劣化决策。

### 12.2 运行时组合（取代 v2.1 的"叠加"说法）

resolver 解析顺序如下，**hub 与 local 不是简单叠加，而是分别贡献不同切面**：

```
resolve(target, stage)
├─ hub.skills/   按 §6.2 selector 命中 domain → 拉 requires → core + technique
├─ hub.knowledge 调 retrieve() 查 target-anchored + mechanism-anchored，取 top-k
└─ local.memory  调 retrieve() 查同一 query，取 top-k（在途、含个人未晋升的 idea）
   → 合并去重（同稳定 ID 以 hub 为准；local 仅补充未晋升的）
   → 按上下文预算裁切（per-stage token cap）
   → 注入 agent context
```

**上下文预算（每 pipeline 阶段）**：

| stage | skills | knowledge top-k | knowledge token cap |
|---|---|---|---|
| research | core 全量 + domain selector 命中 | 8 | 3K |
| plan / plan-review | + technique requires | 5 | 2K |
| implement | + technique requires | 3 | 1.5K |
| code-review | core + 反模式优先 | 5 | 2K |
| test / decision | 仅 anti_pattern + heuristic | 3 | 1K |

数值是起点，按 scorecard 反馈调整。**所有阶段一旦超预算优先丢 `maturity` 低、`score` 低、`evidence` 弱的**。

### 12.3 索引：派生缓存，markdown 为真相源

- **存储**：`hub/knowledge/index/` 与 `local/memory/index/`（faiss 文件或 LanceDB 目录，二选一，对齐 §17 拍板）。Phase 1 用 faiss + sqlite-fts5 起步成本最低，Phase 3+ 评估 LanceDB（单 query 跑混合检索 + scalar 过滤）。
- **增量重嵌（cascade 风格，EverOS 印证）**：watchdog 监 markdown 树，`content_sha256` diff，仅重嵌变更条目；崩溃恢复靠 sqlite 状态队列。**禁止全量重建**作常规路径。
- **重建配方**：每次发布带 `index/manifest.yaml`（embedding model + chunking 参数 + 重建命令），任何成员一行命令重建。

### 12.4 跨工具与版本/降级

- **版本锁定**：`skill-memory.lock`（semver + SHA）固定 hub 版本，等价 package lockfile，可复现、防漂移；每次 run 记录消费版本。
- **故障降级（可用性）**：hub 以 submodule pin 在本地（vendored 副本），**中央仓宕机不阻塞**；resolver 检测不可达即回退上次成功快照并告警。
- **跨工具**：SKILL.md 开放标准，可同时被 OpenCode / Claude Code / Codex 消费，hub 即「团队私有 skill marketplace」。
- **可观测性**：每次 retrieve 记录 `{query, scope, returned_ids, latency, token_used}` 到 `local/runs/<id>/retrieval.jsonl`，喂后续 score 衰减与未被检索条目识别（→ deprecation 候选）。

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
| 读路径可观测 | 每次 retrieve 落 `retrieval.jsonl`；长期未命中条目自动进 deprecation 候选；latency / token_used 直接喂 §14 调优 |

---

## 14. 分阶段路线图

| 阶段 | 周期 | 目标 | 交付物 | 风险 |
|---|---|---|---|---|
| **0 抽取** | 1–2w | 零行为变更跑通双仓 | 切 `skills/agents/pipelines/commands/docs` 到 hub + submodule pin；**路径兼容**（symlink/改写，§6.4）；仓骨架 + schemas + registry | 低 |
| **1 蒸馏 + 读路径 + 本地在线消解** | 3–5w | Tier0→1 结构化 **+ resolver 读路径上线 + 本地 mem0 集成评估** | `hmopt sediment`（含 LLM 显著性 pass，§8）；`memory export` 转标准对象；**`resolver.py` + 混合检索（§12）+ 上下文预算**；**本地 mem0 在线消解集成 PoC**（§10.1.a）：评估 mem0ai 包是否复用 + 自带 UPDATE prompt（绕过 v3 OSS 退化） | 中（mem0 v3 不确定性）|
| **2 策展 + 合并 + 晋升候选检测** | 3–6w | 中央批量合并上线（引擎 A 第二级）| Curator + lint/secret-scan/dedup CI；`policies/` 三文档；**晋升候选自动检测器（§11.5）**：开 promote-candidate PR | 中 |
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
| **mem0 v3 OSS 能力缩水** | OSS 已退化为 ADD-only + hash 去重（§3 提示）；本设计在 §10.1.a 借用 mem0 时**自带 UPDATE/DELETE prompt**（参 v0.1.x 论文版）避免与 OSS 退化耦合；Phase 1 必须显式 PoC 验证 |
| **检索质量退化（新增风险面）** | 加 retrieval 可观测（§12.4）；建小型 retrieval eval 集（"给 query 是否命中预期 ID"），Phase 1 末跑一次基线，发布前回归 |
| **markdown ↔ index 漂移** | cascade 增量重嵌 + content_sha256 校验；每次发布生成 `index/manifest.yaml` 含重建命令；CI 在每次 PR 跑「重建一次索引→对照」校验 |
| **本地消解误删历史事实**（评审反馈 ③）| 七路分类器对 temporal/conditional/selector/evidence 一律不 delete（§10.1.a）；P1-8 PoC 设 **false-delete rate ≈ 0** 硬指标，时态/条件子类单独统计 |
| **subsumption 过度泛化**（评审反馈 ④）| 仅中央 LLM 判定；建链廉价但晋升需 **≥2 个不同被包含实例** + §9 三门（§11.5）；`skills/core/` 候选可要求更高实例门（§17 议题 8）|
| **schema 未收敛即开 Phase 1** | §17 议题 7 升为 P1 前置阻塞；Phase 0.5 DoD 不达成不开工（实现计划 §1）|

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
4. 检索后端：**faiss + sqlite-fts5（Phase 1 起步）vs pgvector（对齐现有 `storage/`）vs LanceDB（一次跑混合检索 + scalar 过滤，EverOS 路线）**。建议 Phase 1 起 faiss，Phase 3 评估 LanceDB。
5. `skills/core/` owner 团队与晋升评审人。
6. **本地在线消解的 mem0 依赖策略**（v2.2 新增）：① 完全自研复刻 v0.1.x 论文 prompt；② 用 `mem0ai` OSS 包拿基础设施 + 自带消解 prompt（避开 v3 退化）；③ 评估 mem0 Platform。决策影响 Phase 1 工期。
7. **markdown 与 schema 落盘格式收敛**（v2.3 升级为 **P1 前置阻塞项**，非普通卫生项）：当前示例（`A001-*.md` 多记录 + 自定义字段）与 `memory_item.schema.json` 不一致；后续 lint / dedup / retrieval scalar filter / Curator 七路分类**全部依赖 schema 字段稳定**，故必须**先于 Phase 1** 收敛。已定方向（§6.1 / §7）：① 一记录一文件 + frontmatter 全 schema 字段；② **文件路径编码 scope** 且 CI 校验路径 scope 与 frontmatter scope 一致；③ `parse_memory.py` 输出标准 schema object，不允许每类自扩字段；④ schema 同时补 `subsumes[]/subsumed_by[]/superseded_by[]/applies_when`。**剩余待拍板**：路径编码粒度（是否到 `targets/<slug>/facts/` 这层）。
8. **subsumption 判定的 LLM 成本与误判**（v2.3 新增）：subsumption 需 LLM 蕴含判定，比 dedup 贵；且过度泛化有风险。已加 ≥2 实例门兜底（§11.5），仍需拍板：中央 Curator 每轮跑 subsumption 的算力预算上限 + 是否对 `skills/core/` 候选要求更高的实例门（如 ≥3）。
