# HMOPT 交互式 Agent 工作台 — 完整设计与使用文档

状态:定稿(融合外部 Workbench v3 设计的最终版)
版本:2.1 · 2026-08-05(v2.0 基础上补全实现思路与使用指南)
读者:HMOPT 维护者、OpenCode 接入工程师、全体团队使用者
配套英文版:`Agent_Workbench_Design_EN.md`

---

## 第一部分 设计

### 1. 问题与核心裁决

**两个真实使用反馈:**

1. 全自动 pipeline 不是大多数人想要的 —— 用户想与每个 agent 直接交互、拿到结构化输出、
   **自己决定下一步交给谁**;
2. agent 过度特定化 —— IC 优化、hotpath 分析焊死在 prompt 里,非优化场景(理解、排查、
   重构)完全不可用;7 个研究 agent 本质是"同一角色 + 不同领域知识"。

**核心裁决:** 重构为**可组合、用户主控的交互式工作台**:

```
日常工作单元 = 一个稳定角色 + 一小组选定技能 + 一个任务工作区(轻量)
```

- 默认入口 `assistant`,普通提问**永不**隐式进入 pipeline;
- 用户拥有路由权:角色切换/咨询/交接全部显式,agent 只建议不抢权;
- 现有全自动 pipeline 保留为兼容配方(coordinator 驱动同一批角色);
- 产品级转变:**从"manager 拥有 pipeline"到"用户拥有任务工作区"**。

实现约束:一切落在 markdown / frontmatter / 文件约定上 —— 不新建服务、不写插件、
不做事件溯源(对外部 Workbench v3 平台级方案的明确取舍,见 §11)。

### 2. 设计规则(黄金规则)

> **责任或权限变了才建角色;领域或方法变了建技能;好用的角色+技能组合重复出现建
> profile;多活动需要可重复编排才建工作流;安全/质量要求需要独立强制才建策略。
> 任务真相存工作区,可复用真相存 Team Memory / Skill Hub。**

反模式(禁止):为"memory-researcher / hotpath-reviewer"建角色(应建 profile);
技能里写"下一步交给谁";技能授予权限;在角色间传递整段对话史;把一次成功对话直接
写进共享技能。

### 3. 分层架构(五层,各管一件事)

```
┌ 任务工作区 workspace/capsule ── 状态层:任务真相在文件里,不在任何 agent 脑中
├ Profile ──────────────────────  复用层:角色+技能命名组合(可选,不是必经层)
├ 技能库 skills(三类)──────────  能力层:领域知识、方法论、怎么干
├ 角色 Roles(7 个)─────────────  责任层:使命/流程骨架/输出契约/权限天花板
└ agent-core 基座契约 ──────────  行为层:所有角色共享的 I/O 契约、交互动词、capsule 维护
```

解耦对照 —— 现有 agent prompt 里焊死的五种东西分别拆到:

| 原耦合内容 | 拆到 | 强制方式 |
|---|---|---|
| 领域知识(reclaim/IC/hotpath) | scenario 技能包 | 角色 prompt 零领域词汇(验收项) |
| 权限边界(散文式"你不要改代码") | 角色 frontmatter `permission` | **OpenCode 运行时真强制** |
| 流程规则(阶段门、下一步给谁) | infra/pipeline 包,仅 coordinator 加载 | 普通角色看不到 stage-gate |
| 任务状态 | workspace/capsule 文件 | 状态在文件,换角色/重开会话不丢 |
| 质量要求 | 工件状态门控(§8.3) | 在状态晋升点把关,不强制流程顺序 |

### 3.5 `.opencode/` 目标总目录结构(与现状对照)

双车道共存:**工作台车道**新增 `agents/`(重建)、`skills/`(重排)、`local/workspaces/`;
**pipeline 车道**的工件与记忆目录全部原位保留。`local/` 是 git-ignored 运行态
(沿用 PR#39 为 sediment_staging 建立的约定)。

```
.opencode/
├── CLAUDE.md                    # M2 改为"薄宪法";pipeline 强制条款移入 pipeline 技能包
├── config.yaml                  # 不变
├── skill-memory.lock            # 不变(hub broadcast 产物)
│
├── agents/                      # ── 责任层(M2 重建)──
│   ├── assistant.md             # 7 个通用角色:assistant / researcher / architect /
│   ├── researcher.md            # implementer / reviewer / validator(mode: all)
│   ├── architect.md             # + coordinator(mode: primary)
│   ├── implementer.md
│   ├── reviewer.md
│   ├── validator.md
│   ├── coordinator.md
│   ├── profiles/                # ── 复用层(M3)── 薄 agent 文件,OpenCode 子目录可发现
│   │   ├── reclaim-investigator.md      # 4 个领域研究 agent 转化而来
│   │   ├── hyperhold-io.md · workqueue.md · sync-mechanism.md
│   │   ├── kernel-understand.md         # 非优化场景(证明通用性)
│   │   └── bug-fix.md
│   └── legacy/                  # M2–M4 过渡别名(hm-opt-manager 等 15 个旧 agent),M4 删除
│
├── skills/                      # ── 能力层(M1 重排,§5)──
│   ├── _registry.yaml
│   ├── role/                    # research-discipline / plan-funnel / review-checklists /
│   │                            # implementation-guardrails / validation-flight-check
│   ├── scenario/
│   │   └── kernel-opt/          # 现有优化技能全家:perf-bottleneck-playbooks / IC /
│   │                            # memory-tlb / ab-test* / iterative / build-and-sign / flash-device
│   └── infra/
│       ├── agent-core/          # §7 基座契约(新)
│       ├── team-memory/ hub-bridge/ language-config/
│       └── pipeline/            # stage-gate + handoff-contract + delegate(仅 coordinator 加载)
│
├── local/                       # ── 状态层(git-ignored 运行态)──
│   ├── workspaces/<task-slug>/  # task.md / capsule.md / artifacts/ / decisions.md(M2,§8)
│   └── sediment_staging/        # team-memory 既有,不动
│
├── memory/                      # 不变:pipeline 车道记忆 + team-memory sediment 源
│   ├── global_lessons.md · targets/ · subsystems/ · human_decisions/ · idea_ledger/
├── state/                       # bad_plans.md 不变;current_task.json M2–M3 作兼容指针,M4 收敛
│
├── commands/                    # /optimize_* 不变(pipeline 配方入口);plan/research 等指向新角色
├── pipelines/                   # 配方卡不变,仅 coordinator + pipeline 包消费
│
├── docs/                        # harness_engineer_system.md M2 起只约束 pipeline 车道;bootstrap 不变
└── bench/ plans/ reviews/ patches/   # pipeline 车道工件目录,原位保留;
                                      # 工作台车道工件写 local/workspaces/<slug>/artifacts/
```

现有 14 项顶层内容的去向速查:**重排** skills/;**重建** agents/(旧件入 legacy/);
**新增** agents/profiles/、skills/infra/agent-core、local/workspaces/;**不变**
memory/、commands/、pipelines/、bench/、plans/、reviews/、patches/、config.yaml、
skill-memory.lock;**修订** CLAUDE.md(薄宪法)、docs/harness_engineer_system.md
(限定 pipeline 车道)、state/current_task.json(M4 收敛)。

### 4. 角色目录(7 个)

规范名 + 别名,`mode: all`(人可直接对话,coordinator 也可委派):

| 角色 | 别名 | 使命(领域无关) | 权限天花板(frontmatter 强制) |
|---|---|---|---|
| `assistant` | — | **默认入口**:答简单问题、做小改动、识别何时值得开工作区、推荐角色/技能 | read: allow · edit: ask · bash: ask |
| `researcher` | 7 个研究变体合一 | 建立可信系统模型:事实/推断/假设分离、证据引用、产出 research-note | read: allow · **edit: deny** · 只读 bash allow |
| `architect` | planner | 从证据到方案:选项生成、权衡、决策记录、产出 plan(5-idea funnel 为可选技能) | 源码 **edit: deny** · plan 工件 write: allow |
| `implementer` | coder | 按已接受 plan 实现:最小 diff、记录假设与偏差;**不自批** | edit: ask(profile 可预批)· 破坏性 deny |
| `reviewer` | plan+code 二合一 | **清洁上下文**独立挑战 research/plan/patch:verdict + 必改项 | **edit: deny** · 只写 review 工件 |
| `validator` | tester | 构建/测试/基准/设备验证声明:区分实现失败/假设失败/基础设施失败 | 操作类 ask · 设备烧写**每次显式批准** |
| `coordinator` | orchestrator(可选) | 仅 pipeline 配方或真并行时启用:分解/委派/汇合;不拥有领域真相、不写源码 | mode: primary · 委派 allow · edit: deny |

角色 prompt 只含:使命、通用流程骨架、输出契约、技能加载规则、权限说明。
**权限天花板是全设计唯一的运行时强制**,OpenCode 原生实现,零新代码。

### 5. 技能库(三层分类 + 注册表 + 四通道加载)

#### 5.1 目录

```
.opencode/skills/
  _registry.yaml     # 唯一注册表(不做 per-skill sidecar)
  role/              # 角色技能:research-discipline / plan-funnel / review-checklists
                     # implementation-guardrails / validation-flight-check
  scenario/          # 场景包:kernel-opt/(现优化全家)· kernel-understand/ · bug-fix/
  infra/             # agent-core / team-memory / hub-bridge / language-config
                     # pipeline/(stage-gate+handoff+delegate,仅 coordinator)
```

#### 5.2 注册表条目(吸收 v3 manifest 的高价值字段)

```yaml
- name: memory-tlb-optimization
  tier: scenario/kernel-opt
  class: optimization-method     # 标签:domain|method|scenario|review|validation|tool|output
  roles: [researcher, architect, reviewer, validator]
  applies_when: ["内存管理 syscall 优化", "TLB/页表路径"]    # 正触发
  not_for: ["纯解释类任务", "无内存证据的延迟声明"]           # 负触发,降误载
  conflicts: []
  context_cost: ~400 lines       # 超 500 行必须拆 references/
  risk: R0                       # R0 只读 R1 写文档 R2 改源码/构建 R3 设备/发布
```

#### 5.3 加载:四条通道 + 三个纪律

```
优先级:① 用户显式 > ② profile 预载 > ③ 触发建议(suggest,人确认)> ④ 角色默认
```

纪律:**渐进披露**(注册表常驻仅 name+description+applies_when 一行,~80 token/技能,
命中才载全文)· **活跃非核心技能 ≤4**、触发推荐 **≤3 且说明理由** · **composition
receipt**(§8.4)反馈闭环修触发词。

### 6. Profiles(复用层)

Profile = 角色 + 默认技能 + 可选技能 + 权限偏好的**命名组合**,解决"我要定制 agent"
的 90% 需求而不新增角色。落地为薄 agent 文件(OpenCode 可 Tab/@ 直选):

```markdown
--- # .opencode/profiles/reclaim-investigator.md
description: reclaim 子系统调查员(researcher + reclaim 领域包)
mode: primary
base_role: researcher
skills: [research-discipline, kernel-opt/perf-bottleneck-playbooks, kernel-opt/domain-reclaim]
optional_skills: [kernel-opt/memory-tlb-optimization]
---
按 researcher 角色契约工作,已预载 reclaim 领域上下文。
```

作用域覆盖:团队策展(hub)< 项目仓库 < 个人配置 < 会话显式;权限不在覆盖链内。
现有 4 个领域研究 agent → 4 个 profile。

### 7. 交互模型

#### 7.1 六个交互动词(人是所有权仲裁者)

| 动词 | 会话所有权 | 说明 |
|---|---|---|
| continue | 当前角色 | 同一责任继续 |
| add/remove skill | 当前角色 | 方法变、责任不变 |
| **consult** | **不转移** | 有界咨询:@目标角色单发,拿紧凑结论回来 |
| **handoff** | 转移 | 责任/权限边界变了:转发(可编辑)brief,目标角色成对话主 |
| fork | 新分支 | 复制 capsule 开分支,比较备选,不覆写原状态 |
| recipe | coordinator | 显式启动 pipeline 配方(`/optimize_*`) |

#### 7.2 每回合输出契约(role report)

① 身份横幅 → ② 结构化结果 → ③ 写盘工件清单 → ④ **Next options(1~3 条:动词+目标
角色+理由+可直接转发的 brief 草稿)** → ⑤ 开放问题与置信度 → ⑥ capsule 更新。
interactive/guided 模式下必须等用户选择,永不自动转移。

#### 7.3 清洁上下文评审

reviewer 收到的是**需求+工件+证据+决策记录**,不是 implementer 的自我陈述;consult
模式(subagent 新上下文)天然满足。防作者偏见污染评审。

#### 7.4 多 Agent 资格闸

并行前必须同时满足:≥2 条真正独立分支 · 共享可变状态最小 · 每分支 IO 明确 · 有汇合
规则 · 有预算 · 有"单角色+技能不够"的可度量理由。否则单角色+技能。

### 8. 任务工作区(轻量文件化)

#### 8.1 布局

```
.opencode/local/workspaces/<task-slug>/
  task.md          # 目标/范围/约束/状态(ready|running|waiting-user|done)
  capsule.md       # Task Capsule:当前投影(核心工件,见下)
  artifacts/       # research-note.md / plan.md / review.md / validation.md …
  decisions.md     # 决策与被拒选项(追加式)
```

取代 singleton `current_task.json` 的长期方向;fork = 复制目录。
不做:events.jsonl、snapshots、乐观并发、SQLite。

#### 8.2 Task Capsule

```markdown
# Capsule: reclaim-race 调查
objective: 诊断 shrink_node 竞态
scope: mm/vmscan.c · commit abc1234 · 符号 shrink_node
constraints: [保持 ABI, 设备操作需批准]
active: researcher + [research-discipline, domain-reclaim] · mode: guided
confirmed_facts:
  - 锁 X 覆盖回调 Y(evidence: vmscan.c:137-155)
open_questions: [回调 Y 是否允许睡眠?]
decisions: [先测竞争度再考虑拆锁]
artifacts: [artifacts/research-note.md]
```

当前角色每回合末更新(agent-core 契约必填);**handoff/consult 传 capsule + 工件引用,
不传对话史**;compaction 后只重注入 capsule —— 一份文件解决交接、恢复、压缩三个问题。

#### 8.3 工件状态门控(管状态,不管流程)

工件头部 `status: draft | reviewed | approved | validated | superseded`。draft 自由
产出;**状态晋升有条件**:patch→`ready-to-land` 需 review approved(+构建过);性能
声明→`validated` 需可比 A/B 证据(基线+候选+指标匹配+噪声底)。修正生成新版本,旧版
标 superseded。由 reviewer/validator 在角色层执行,不建 Policy Engine。

#### 8.4 Composition receipt

每个工件头一行:`produced_by: researcher + [domain-reclaim, method-data-flow] · 2026-08-05`。
一行成本,换 hub 侧"哪些技能组合产出被接受工件"的策展证据流。

### 9. Pipeline 兼容

- `/optimize_*` 保留 = coordinator + pipeline 技能包驱动同一批角色;stage-gate 只在该包内;
- 老 agent 保留 alias 至 M4;回归清单 = 4 条 optimize 命令 golden 用例;
- Python/LangGraph 编排本设计不动(整体推迟评估)。

### 10. 与 Team Memory / Skill Hub 集成

- agent-core 把 memory_recall/log/feedback 定为所有角色通用行为;
- receipt + 工件接受结果 → hub 技能策展证据流;
- 四层分离:任务态(工作区)≠ 检查点(capsule)≠ 个人经验(journal)≠ 团队策展(hub);
  一次成功对话永不自动改写共享技能。

### 11. 对外部 Workbench v3 的吸收/拒绝决策记录

**吸收(12 项,全部降级为文件/约定级)**:用户拥有工作区框架 · assistant 默认角色 ·
黄金规则 · Task Capsule · 六交互动词 · 清洁上下文评审 · 多 agent 资格闸 · 权限天花板
+R0-R3 标签 · 工件状态门控 · profile 机制 · 技能选择数字纪律与 applies_when/not_for ·
composition receipt 与非优化场景进 MVP。

**拒绝(规模不匹配)**:事件溯源工作区 · Capability Broker 与逻辑能力词表 · Policy
Engine 服务 · Workflow 编译器与条件语言 · Python/LangGraph 契约统一 · workbench TS
插件 · 17 个版本化 JSON schema 运行时校验 · 全套 /task /consult 命令(OpenCode 原生够)·
Composition Lock 全量钉版。

---

## 第二部分 实现思路

### 12. 机制 → 实现载体对照(整个设计怎么落地)

| 机制 | 实现载体 | 新代码量 |
|---|---|---|
| 角色与权限天花板 | 7 个 agent markdown + frontmatter `permission` | 0(纯配置) |
| agent-core 契约 | 1 个 infra 技能文件(输出契约/动词/capsule 规则/状态门控/资格闸) | 0 |
| 技能三层与触发 | 目录重排 + `_registry.yaml` | 1 个 lint 脚本(注册表校验) |
| profile | 薄 agent 文件若干 | 0 |
| 工作区/capsule | 目录约定 + 模板文件 | 0(可选:初始化脚本) |
| 状态门控/receipt | 工件头部约定 + reviewer/validator 契约条款 | 0 |
| pipeline 兼容 | coordinator(manager 瘦身)+ pipeline 技能包 | 改写 1 个 agent |

全设计新增可执行代码 ≈ 一个注册表 lint 脚本;其余全部是 markdown/配置。

### 13. 迁移计划(M1–M4,每期独立交付)

| 期 | 内容 | DoD |
|---|---|---|
| **M1** 技能库重组 | 三层目录重排、`_registry.yaml`(含 applies_when/not_for/risk/cost)、技能瘦身审计(≤500 行)、引用路径更新 | 4 条 optimize 命令回归通过;注册表 lint 绿;**零行为变化** |
| **M2** 角色+基座+工作区 | agent-core 契约技能;7 角色文件(权限 frontmatter);工作区/capsule 模板;旧 agent 留 alias | 人为路由端到端跑通真实任务(assistant→researcher→consult reviewer→architect→implementer→reviewer→validator),全程 capsule 交接;researcher 尝试 edit 被运行时拒绝 |
| **M3** profile 化+场景扩展 | 4 领域 agent→4 profile;kernel-understand、bug-fix 两个非优化 profile;receipt 落地;suggest 策略上线 | 同一 researcher 仅换技能跑通 ≥3 领域;非优化任务全程无优化词汇 |
| **M4** coordinator+收尾 | manager 重构为 coordinator;`/optimize_*` 指向新链路;删旧 agent;使用指南发布 | 4 条 optimize 命令 golden 回归通过 |

### 14. 验证方法

- **回归**:M1 前先固化 4 条 optimize 命令的 golden 输出(工件清单+关键字段),每期跑;
- **对比**:M2 用同一任务对比新旧 researcher 产物质量(领域深度不下降);
- **权限**:每角色一条"越界尝试"用例(researcher 试图 edit / reviewer 试图改源码);
- **token**:M1 前后量基础上下文 token(渐进披露收益量化);
- **触发**:注册表 applies_when/not_for 用 10 个正例 + 10 个负例任务描述测准确率。

---

## 第三部分 使用指南

### 15. 快速上手:三种进入方式

1. **直接提问**(默认 assistant):打开 OpenCode 直接说话 —— 简单问题当场答,复杂任务
   assistant 会建议"开个工作区 + 找 researcher",**你确认才走**;
2. **直选角色**:Tab 切换或 `@researcher` 直接找某角色;
3. **用 profile**:`@reclaim-investigator` 一步进入"researcher+领域包"的预组合。

### 16. 典型场景

**场景 A:普通问题(不开工作区)**

```
你:shrink_node 里这个 sc->priority 是干嘛的?
assistant:(直接解释,给 file:line)…… 这是个简单问题,无需开任务。
```

**场景 B:深入调查(单角色 + 技能建议)**

```
你:@researcher 帮我查 shrink_node 的竞态嫌疑
researcher:=== researcher — acknowledging: shrink_node 竞态 ===
  已开工作区 workspaces/shrink-node-race/。
  建议加载(请确认):domain-reclaim(路径命中 mm/)· method-lifetime(关键词"竞态")
你:好
researcher:(调查……)
  ## 结果:锁 X 覆盖回调 Y,但 Z 路径存在窗口(evidence: vmscan.c:137-155)
  工件:artifacts/research-note.md(status: draft)
  ## Next options
  1. consult reviewer — 独立挑战竞态分析(brief 草稿:附 note+capsule)
  2. handoff architect — 基于结论出修复选项
  3. continue — 把 Z 路径的睡眠语义查清
你:1
```

**场景 C:完整变更流(人为路由,brief 转发)**

```
researcher 产出 note → 你转发建议 brief 给 architect
architect:两个选项(A 拆锁 / B 延迟处理),权衡表,决策记录 → plan.md(draft)
你:选 A,handoff implementer
implementer:(请求 edit 批准)→ patch + 自述正确性 → 建议 review
你:consult reviewer(清洁上下文,只带 plan+patch+证据)
reviewer:verdict: approved,patch 状态可晋升 ready-to-land(构建过后)
你:handoff validator → lmbench A/B → validation.md,声明 validated
```

每步你都可以:改 brief 再转发、要求重做、加/减技能、或直接停(任何工件都可即停即用)。

**场景 D:fork 比较方案**

```
你:fork 一下,另一支试 B 方案
→ workspaces/shrink-node-race-b/(capsule 复制),两支各自推进,最后对比 validation
```

**场景 E:老全自动 pipeline(不变)**

```
/optimize_workqueue        # coordinator 驱动全流程,阶段门强制,和从前一样
```

### 17. 用户侧速查

| 想做什么 | 怎么做 |
|---|---|
| 换角色 | Tab 或 `@角色名`,或采纳 Next options 里的 handoff 建议 |
| 加/减技能 | "加载 memory-tlb" / "卸掉 IC 技能" |
| 问为什么推荐某技能 | "为什么建议这个技能?"(注册表触发理由) |
| 独立评审但不换主对话 | consult reviewer(评审完回到你手上) |
| 保存现场下次继续 | 工作区自动持久;重开会话说"继续 shrink-node-race" |
| 比较两个方案 | fork |
| 跑老全自动优化 | `/optimize_*` |

### 18. 用户纪律(Do / Don't)

- **Do**:让 reviewer 走 consult(清洁上下文);工件状态没到 validated 别当结论引用;
  重要决策让 agent 记进 decisions.md;好用的组合让团队沉淀成 profile。
- **Don't**:别把对话史整段贴给下一个角色(转 capsule+工件);别绕过 implementer 的
  edit 批准;别要求 researcher 直接改代码(权限会拒,这是设计而非故障)。

---

## 第四部分 验收与风险

### 19. 验收标准

- **模块性**:核心角色 prompt 无子系统路径、无 IC 假设;领域技能 ≥3 角色可复用;
- **用户主控**:普通提问永不隐式进 pipeline;转移必经用户;可查技能推荐理由;
- **权限**:技能不能扩权;researcher/reviewer 的 edit:deny 所有测试成立;R3 每次显式批准;
- **状态**:任务可恢复(capsule 载入即续);fork 不覆写;
- **兼容**:4 条 optimize 命令全绿;
- **效率**:渐进披露后基础上下文 token 下降(量化);多数普通任务单角色完成。

### 20. 风险与对策

| 风险 | 对策 |
|---|---|
| pipeline 回归(M4 最大) | 单独一期;alias 并行;golden 先建后切 |
| assistant 长成新 god-manager | 权限 edit:ask + 契约明写"识别并移交,不自己扛" |
| 技能误载/漏载 | applies_when/not_for 双向触发 + suggest 人确认 + receipt 反馈修 description |
| capsule 维护被遗忘 | agent-core 设为输出契约必填;reviewer 检查 |
| 角色抽象后领域深度下降 | 深度只搬家不丢失;M2 新旧对比验证 |
| 工作区与 current_task.json 双轨 | M2-M3 兼容指针,M4 收敛 |
