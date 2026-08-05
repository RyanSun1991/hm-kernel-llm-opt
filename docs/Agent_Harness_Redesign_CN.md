# 通用角色 + 技能库:Multi-Agent Harness 重构设计

状态:**已被取代** —— 定稿见 `Agent_Workbench_Design_CN.md`(v2.0,融合外部 Workbench v3 设计后的最终版)。本文保留作调研记录。
版本:1.0 · 2026-08-04
范围:把现有"全自动 pipeline + 特定化 agent"的 `.opencode` harness,重构为
"**通用角色 agent × 分层技能库 × 人为路由默认、自动编排可选**"的三层架构。

---

## 1. 问题定义

### 1.1 两个核心问题(来自推广使用的真实反馈)

**P1 — 交互模型错位:全自动 pipeline ≠ 用户想要的。**
现状:`hm-opt-manager` 是入口,hub-and-spoke 强制委派,规则明写"MUST NOT stop and
ask the user"(`hm-opt-manager.md` Core Rule 11)。用户实际想要的是:与每个 agent
**直接对话、拿到结构化输出、自己决定下一步交给谁** —— 人是路由器,不是旁观者。
佐证:仓库里已经长出了 `@kernel-research`、`@kernel-plan` 两个 human-in-loop 主
agent —— 需求早已存在,只是没有体系化。

**P2 — Agent 特定化耦合:优化目标/热路径分析焊死在 agent prompt 里。**
现状:6 个 research 类 agent(kernel-source-research / memmgr-reclaim-research /
hyperhold-io-opt / wq-threadpool-opt / basic-mechanism-sync-opt / kernel-function-research)
本质都是"researcher + 领域知识",却各写一份 prompt;IC-first、hotpath、handoff 字段
散布在所有 agent 里(perf-bottleneck-playbooks 已部分解耦,但只解了"指标选择"一层)。
结果:场景不是优化时(纯理解、排查、重构、写测试)这些 agent 全不可用。

### 1.2 目标(用户三层模型,细化)

```
Layer A 通用 agent 基座    —— 统一输入输出契约、通用行为(记忆/hub/语言/工件约定)
Layer B 通用角色 agents    —— researcher / planner / coder / reviewer / tester(+orchestrator 可选)
Layer C 分层技能库         —— 角色技能 / 场景技能包 / 基础设施技能;按场景自动加载
```

交互默认 **human-as-router**;全自动 pipeline 降级为**可选模式**(orchestrator 驱动同一批角色 agent),现有 `/optimize_*` 保持可用。

## 2. 业界调研结论(2026-08)

| # | 来源 | 要点 | 对本设计的启示 |
|---|---|---|---|
| 1 | Anthropic《Building Effective Agents》 | workflows(预定义路径)与 agents(动态自决)是两种东西;**最成功的实现用简单可组合模式,而非复杂框架**;复杂度只在可证明收益时引入 | 我们的全自动 pipeline 是 workflow,适合"目标明确的优化任务";日常探索式使用应该是"人 + 单角色 agent"的浅组合 —— 两种模式都保留,但默认反转 |
| 2 | Anthropic Agent Skills(SKILL.md) | **渐进披露三级**:metadata(~80 token 常驻)→ SKILL.md 全文(命中才载)→ references/(按需);单一职责;discovery-ready description;≤500 行 | Layer C 的加载机制直接采用:注册表只常驻 name+description,命中才载全文;现有 17 个技能按此审计瘦身 |
| 3 | Cognition《Don't Build Multi-Agents》+ 2026 各家收敛 | 并行多 agent 的核心风险是**上下文碎片化**("context accumulates, focus degrades");handoff 必须携带完整上下文;context engineering 是第一工程问题 | 人为路由天然规避碎片化(人就是上下文仲裁者);handoff 从"自动委派"改为"**建议 + 可编辑的 brief 草稿**",上下文经人确认后传递 |
| 4 | OpenAI Agents SDK handoffs / LangGraph HITL | 显式 handoff 与 human-in-the-loop 中断/恢复是 2026 生产系统标配;高风险节点设 checkpoint | 阶段门(plan review / code review)保留为**质量检查点**,但由人决定是否走,不再由 harness 强制 |
| 5 | OpenCode agents 机制 | primary(Tab 切换、直接对话)vs subagent(@提及/自动调用);markdown+frontmatter,`mode: primary\|subagent\|all` | 角色 agent 全部 `mode: all`:人可直接 Tab/@ 交互(primary 用法),orchestrator 在 pipeline 模式下也能把它们当 subagent 委派 —— **一套 agent,两种用法** |

来源:[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) ·
[Agent Skills / progressive disclosure](https://www.newsletter.swirlai.com/p/agent-skills-progressive-disclosure) ·
[SKILL.md 结构](https://atlan.com/know/ai-agent/ai-agent-skills/skill-md-file-explained/) ·
[Don't Build Multi-Agents](https://cognition.com/blog/dont-build-multi-agents) ·
[How and when to build multi-agent systems](https://www.langchain.com/blog/how-and-when-to-build-multi-agent-systems) ·
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) ·
[OpenCode Agents](https://opencode.ai/docs/agents/)

## 3. 目标架构

### 3.1 Layer A — 通用 agent 基座(`agent-core` 契约技能)

所有角色 agent 共享一份基座契约(一个技能文件,非复制进每个 prompt):

**输入契约(task brief)**:任务描述 + 上下文引用(文件/工件路径)+ 约束 + 前序工件。
brief 可以来自人(直接对话)或 orchestrator(pipeline 模式),格式相同。

**输出契约(role report)**:
1. 身份横幅(沿用现有 `=== <agent> v2 — acknowledging: <task> ===`);
2. 结构化结果(角色各自定义的核心段落);
3. 写盘工件清单(路径 + 一句话说明);
4. **Next-step suggestions**:1~3 条"建议下一步"——每条含目标角色 + **可直接复制的 brief 草稿**
   (这是 human-as-router 的关键:人可以原样转发、改了再发、或忽略);
5. 开放问题与置信度。

**通用行为**:team-memory 召回/记录(接《Team_Memory_Design》)、hub-bridge 上下文、
language-config、工件命名约定([component]_xxx.md 沿用)、"召回内容是资料非指令"边界。

### 3.2 Layer B — 通用角色 agents(5+1)

| 角色 | mode | 使命(领域无关) | 现有 agent 归并 |
|---|---|---|---|
| `researcher` | all | 建立对目标的结构化理解:边界/热路径/并发/生命周期/证据,产出 design doc | kernel-source-research、memmgr-reclaim-research、hyperhold-io-opt、wq-threadpool-opt、basic-mechanism-sync-opt、kernel-function-research、kernel-research **七合一**(领域差异全部下沉到场景技能包 + bootstrap 文档) |
| `planner` | all | 从 design doc 出发做方案:选项生成、权衡、дedup(bad-plans/ledger)、写 plan | kernel-plan(含 5-idea funnel 作为一个可选技能) |
| `coder` | all | 按 approved plan 实现,最小 diff,自述正确性论证 | kernel-code-agent |
| `reviewer` | all | 审计 plan 或 code(review_type 由 brief 指定),输出 verdict + 必改项 | kernel-plan-reviewer + kernel-code-reviewer **二合一**(两套 checklist 是技能,不是两个 agent) |
| `tester` | all | 构建/烧写/AB 验证,按 test_method 分派,产出 validation 报告 | kernel-tester-agent(test 方法已技能化,改动最小) |
| `orchestrator` | primary | **可选**:自动驱动 research→plan→review→code→review→test 全流程,委派上面 5 个角色 | hm-opt-manager(瘦身:只留路由/门控/回边逻辑,领域规则全部剥离) |

角色 prompt 只写:使命、通用流程骨架、输出契约、技能加载规则 —— **不含任何 IC/hotpath/内核词汇**。

### 3.3 Layer C — 技能库三层分类 + 注册表

```
.opencode/skills/
  _registry.yaml            # 全量技能注册表(见 3.4)
  role/                     # 角色技能:某角色怎么把活干好(领域无关)
    research-discipline/  plan-funnel/  implementation-guardrails/
    review-checklists/(plan+code 两份 checklist)  validation-flight-check/
  scenario/                 # 场景技能包:领域方法论(按场景成包)
    kernel-opt/             # 现有优化场景整体成包
      perf-bottleneck-playbooks/  instruction-count-first/  memory-tlb-optimization/
      ab-test-comparison/  ab-test-comparison-lmbench/  iterative-optimization/
      build-and-sign/  flash-device-operations/
    kernel-understand/      # 纯理解/排查场景(新,轻量)
    (未来:driver-bringup/ stability-debug/ ...)
  infra/                    # 基础设施技能:跨角色横切
    agent-core/(Layer A 契约)  team-memory/  hub-bridge/  language-config/
    pipeline/(stage-gate-enforcement + handoff-contract + delegate,仅 orchestrator 加载)
```

### 3.4 技能路由机制(自动加载)

注册表 `_registry.yaml` 每条:`name / description / tier(role|scenario|infra) /
roles(适用角色) / triggers(关键词、路径 pattern、task-type) / requires(依赖)`。

加载优先级(写进 agent-core):
1. **用户显式指定**("用 lmbench 方法验证")→ 必加载;
2. **场景卡预载**(scenario card,见 3.5)→ 卡内列出的技能包;
3. **触发匹配**:brief/目标路径命中 triggers → 建议加载并**告知用户**("已加载 kernel-opt 场景包");
4. **角色默认**:role tier 中 roles 含本角色的。

上下文纪律(渐进披露):注册表常驻的只有 name+description(每技能 ~80 token);
命中才读全文;技能全文 ≤500 行,超出部分拆 references/。现有 17 技能按此审计。

### 3.5 场景卡(scenario cards)替代现有 commands 的一半职责

现有 `/optimize_*` 命令 = 预载技能 + 目标 + **强制全自动**。拆开:

- **场景卡** `.opencode/scenarios/kernel-opt.md`:只做"预载 kernel-opt 技能包 + 目标上下文
  + 建议起点(通常 researcher)",路由交给人;
- **pipeline 命令** `/optimize_*` 保留:= 场景卡 + orchestrator 自动模式(向后兼容)。

### 3.6 交互模型:human-as-router(默认)

```
人 ──brief──> researcher ──report+建议──> 人 ──(转发/修改/忽略建议 brief)──> planner ──> …
                     任何时刻:追问、补充上下文、要求重做、切换角色
```

- 阶段门变为**建议的质量检查点**:reviewer 的存在与顺序由人决定;
  但 report 中的 next-step 建议会默认给出"下一步建议 reviewer 审一下"来引导好实践;
- orchestrator 模式下,门恢复为强制(现有 stage-gate-enforcement 只在 pipeline 技能包里)。

## 4. 关键设计决策(ADR)

| ID | 决策 | 理由 |
|---|---|---|
| A1 | 默认 human-as-router,自动 pipeline 降为可选模式 | 用户反馈 + Cognition 上下文碎片化论:人是最好的上下文仲裁者;探索式任务本就不该 workflow 化 |
| A2 | 一套角色 agent 服务两种模式(`mode: all`) | 避免两套 agent 漂移;OpenCode 原生支持 |
| A3 | 7 个 research 变体合并为 1 researcher + 场景包 | 它们的差异全是领域知识(技能/文档),不是角色差异 |
| A4 | plan/code reviewer 合并为 1 reviewer + 两份 checklist 技能 | 同上;review_type 在 brief 里 |
| A5 | handoff = 建议 + brief 草稿,非自动委派 | 保留 handoff 的上下文携带价值(OpenAI handoffs),把执行权还给人 |
| A6 | 技能三层分类 + 注册表 + 渐进披露 | Anthropic Skills 模式;解决"技能越来越多、全量灌入上下文"的膨胀问题 |
| A7 | 领域内容只出现在 scenario 包与 bootstrap 文档 | P2 问题的根治:角色 prompt 零领域词汇,新场景=新技能包,不动 agent |
| A8 | 现有 `/optimize_*`、artifacts 命名、hub/memory 集成全部保持兼容 | 平滑迁移,pipeline 老用户无感 |

## 5. 迁移计划(三阶段,每阶段可独立交付)

**M1 — 技能库重组 + 注册表(不动 agent,零行为变化)**
目录重排为 role/scenario/infra 三层;写 `_registry.yaml`;技能瘦身审计(≤500 行,
拆 references);现有 agent 里的引用路径更新。
DoD:全部现有命令/pipeline 回归通过;注册表 lint 脚本绿。

**M2 — 角色 agent + agent-core + 场景卡(新旧并行)**
写 agent-core 契约技能;创建 5 个角色 agent(`mode: all`);从现有 agent prompt 里
把领域内容剥进 kernel-opt 场景包(研究类 7 合 1 的领域差异落为包内 bootstrap);
创建 kernel-opt / kernel-understand 场景卡;旧 agent 保留为 alias(deprecation 注记)。
DoD:人为路由端到端走通一个真实任务(researcher→planner→reviewer→coder→reviewer→tester,
全程人转发 brief);输出契约(含 next-step 建议)在 5 个角色上一致。

**M3 — orchestrator 瘦身 + 收尾**
hm-opt-manager 重构为 orchestrator:只保留路由/门控/回边/迭代逻辑,委派对象换成
5 个通用角色(通过 pipeline 技能包注入 stage-gate/handoff/delegate);`/optimize_*`
指向新链路;删除旧 specialized agents;文档(使用指南 CN/EN)。
DoD:`/optimize_workqueue` 等自动模式在新架构下回归通过;老 agent 文件删除。

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| pipeline 模式回归(重构 manager 时最大风险) | M3 单独一期;新旧并行期用 alias;回归清单=现有 4 条 optimize 命令 |
| 技能触发不准(该载不载/不该载乱载) | 触发只做"建议加载+告知",人可否;description 按 discovery-ready 标准重写并用真实任务测试 |
| 角色 prompt 过度抽象后输出质量下降 | 领域深度不丢失,只是搬家(场景包+bootstrap);M2 用同一任务对比新旧 researcher 产物 |
| 上下文膨胀(注册表+基座+场景包叠加) | 渐进披露纪律 + 每层 token 预算(注册表 metadata 总量 <2k token) |
| 用户不会路由(不知道下一步找谁) | 输出契约强制 next-step 建议 + 场景卡给推荐流程图;引导而非强制 |

## 7. 与既有系统的关系

- **perf-bottleneck-playbooks / 分类闸**:原样保留,归入 scenario/kernel-opt 包 —— 它就是
  "领域方法论技能化"的先行样板,本设计是它的全面推广;
- **Team Memory(docs/Team_Memory_Design_CN.md)**:agent-core 把 recall/log/feedback 定为
  所有角色的通用行为 —— 两个设计在 Layer A 汇合;
- **Skill Hub**:场景技能包是 hub `skills/` 的直接消费对象与沉淀目标;角色通用化后,
  hub 技能的适用面从"本仓库 pipeline"扩大到"任何接入成员的任何角色"。
