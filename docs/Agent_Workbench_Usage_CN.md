# HMOPT Agent 工作台 — 使用指南

状态：随 M4 收尾发布 · 2026-08-05
设计文档：`Agent_Workbench_Design_CN.md` · 英文版：`Agent_Workbench_Usage_EN.md`

`.opencode/` 系统现在是一个**可组合、由用户掌控的交互式工作台**：每次与一个角色对话，
领域知识由技能包承载，任务状态放在轻量工作区里，**每一次流转都由你决定**。原有的
自动化流水线仍然保留——作为需要你显式启动的配方（recipe）。

---

## 1. 快速上手 — 三条入口

1. **直接提问。** 打开 OpenCode 输入即可。默认 agent 是 `assistant`：简单问题直接给出
   带 `file:line` 证据的回答；复杂任务它会提出"开一个工作区 + 请 researcher 带这些技能
   进场？"——在你确认之前什么都不会发生。
2. **选一个角色。** Tab 切换或 `@researcher`、`@architect`、`@implementer`、
   `@reviewer`、`@validator`。角色本身不含领域知识；每个角色会从注册表给出匹配的技能
   建议（附命中原因），等你确认。
3. **用一个 profile。** `@reclaim-investigator`、`@hyperhold-io`、`@workqueue`、
   `@sync-mechanism`、`@kernel-understand`、`@bug-fix` —— 预装好"角色+技能"的组合，
   跳过建议环节直接开工。

## 2. 角色阵容

| 角色 | 职责 | 硬性上限（运行时强制） |
|---|---|---|
| `assistant` | 默认入口；直接回答、小改动、识别并推荐角色 | 编辑/bash/consult 均需批准 |
| `researcher` | 建立系统模型：事实/推断/假设分离，证据引用 | **不能改源码** |
| `architect` | 备选方案 + 取舍 + 决策记录 + 带验收标准的计划 | **不能改源码** |
| `implementer` | 按已批准计划做最小 diff，记录偏差，从不自我批准 | 每次编辑需批准；破坏性操作拒绝 |
| `reviewer` | 干净上下文中的独立评审：结论 + 必改项 | **不能改源码，也不能改被评审的产物** —— 只能写评审产物与工作区 |
| `validator` | 用构建/测试/基准/设备 A/B 证实或证伪主张 | 变更类命令需批准；设备操作逐次批准 |
| `coordinator` | 仅用于流水线配方与真正的并行工作 | 不能改源码；唯一可 delegate 的角色 |

每个上限都是 frontmatter 里按路径模式限定的 `edit` 映射：角色自己的产物目录可写
（工作区，以及 researcher 的 `docs/`+`memory/`、architect 的 `plans/`、reviewer 的
`reviews/`、validator 的 `bench/`、coordinator 的 `state/`），**其余一律拒绝 ——
首先是源码**；只读 shell 命令自由执行，变更类命令逐条审批。Profile 继承其基础角色
的上限。权限拒绝（比如 researcher 想改源文件被拒）**是设计行为，不是 bug** ——
角色会告诉你谁有权做，并附上可直接转发的任务简报。

一个诚实的注记：设备操作（刷机、真机测试）走 MCP 工具，frontmatter 权限并不拦截
MCP 调用 —— 其逐次审批是 validator 必须宣告并遵守的契约义务（每个设备动作都先问、
每次都问），而非运行时保证。请在下面的运行时清单中验证。

## 3. 技能 — 加载机制

- `.opencode/skills/_registry.yaml` 是唯一索引（role/ · scenario/ · infra/ 三层）。
- 角色开工时只读注册表元数据（每技能约 80 token），将你的任务描述与
  `applies_when` / `not_for` 匹配，**给出 ≤3 条带理由的建议**，你确认后才读取
  SKILL.md 全文。非核心技能同时激活不超过 4 个。
- 问"为什么建议这个技能？"会得到命中的注册表触发词。也可手动"加载 memory-tlb"/
  "卸掉 IC 技能"。
- 技能永远不会放大权限：超出角色上限的技能会降级为仅供参考、逐操作审批、或建议
  handoff 给有权限的角色。

## 4. 任务工作区

超过一轮的任务，角色会开（或你要求开）一个工作区：

```
.opencode/local/workspaces/<task-slug>/   （git 忽略；模板在 .opencode/templates/workspace/）
  task.md        目标 · 范围 · 约束 · 状态
  capsule.md     任务胶囊 —— 交接/恢复的唯一载体
  artifacts/     research-note.md · plan.md · review-*.md · validation.md …
  decisions.md   只追加的决策与被否方案记录
```

- `bash scripts/new_workspace.sh <slug>` 新建；`--fork <源slug>` 复制一个分支做
  对比（fork 永不覆盖原工作区）。
- 当前角色每轮结束都更新 `capsule.md` —— 所以你可以关掉会话，之后说
  **"continue <task-slug>"**：角色重新加载胶囊，恢复目标、状态、悬而未决的问题
  和下一步。
- 交接与咨询传递的是**胶囊 + 产物引用，绝不传聊天记录**。

## 5. 工作流转 — 六个动词

| 动词 | 效果 |
|---|---|
| `continue` | 当前角色继续 |
| `add/remove skill` | 换方法，不换责任 |
| `consult` | 向另一角色发起**干净上下文**的一次性提问；结论回到你手里，主导权不转移 |
| `handoff` | 责任转移；你转发（可先编辑）角色起草的简报 |
| `fork` | 复制工作区，并行比较备选方案 |
| `recipe` | 你显式启动自动化流水线（`/optimize_*`） |

每个角色回合都以 **Next options** 结束 —— 1–3 条建议，各含动词、目标角色、理由和
可直接转发的简报草稿。你不选择，什么都不会执行。

## 6. 产物状态 — 谁有权宣称什么

产物头部携带 `status: draft → reviewed → approved → validated`（另有
`superseded`）和 `produced_by:` 回执（角色 + 技能 + 日期）。状态晋升是有条件的：

- 计划 → `approved`：需要 reviewer 的批准结论
- 补丁 → `ready-to-land`：需要通过的代码评审 **加** 构建通过
- 性能主张 → `validated`：需要 validator 的 A/B 证据（基线 + 候选 + 同指标 + 噪声下限）

不要把 `draft` 当结论引用；先做能让它晋升的 consult / 验证。

## 7. 典型场景

**深入调查**
`@researcher 调查 shrink_node 疑似竞态` → 开工作区 → 技能建议（domain-reclaim：
路径命中；domain-sync：关键词"race"）→ 确认 → 带证据的研究笔记 → Next options：
consult reviewer / handoff architect / continue。

**完整改动（人肉路由）**
researcher 笔记 → 你把简报转给 `@architect` → 方案+取舍表+计划（draft）→ consult
`@reviewer`（干净上下文）→ 批准 → handoff `@implementer`（编辑需批准）→ 实现说明 →
consult reviewer（代码）→ handoff `@validator` → A/B 证据 → 主张 `validated`。

**仅理解** — `@kernel-understand hp_iotab 槽位复用是怎么工作的？`（零优化词汇，
带 file:line 的走读讲解）。

**Bug 诊断** — `@bug-fix 回收压力下偶发挂死` → 固定复现/触发条件 → 机制级诊断 →
handoff implementer 做最小修复。

**自动化流水线（入口不变）**
`/optimize_workqueue`、`/optimize_generic`、`/optimize_hyperhold`、
`/optimize_memmgr_reclaim` —— M4 起由 `@coordinator` 驱动同一批角色
（researcher/reviewer/implementer/validator），在流水线技能包的阶段门下运行：
计划评审 GATE → 代码评审 GATE → 测试 A/B。`Auto-Iterate: N` 照旧。旧链
（`@hm-opt-manager`，`agents/legacy/`）在真实对比运行归档前保留为回退通道 ——
新链行为异常时用同一命令体、首行换成旧 agent 引用即可，并请上报差异。

## 8. 速查表

| 想要… | 做法 |
|---|---|
| 切换角色 | Tab 或 `@角色`，或采纳 Next options 里的 handoff |
| 加/卸技能 | "加载 memory-tlb" / "卸掉 IC 技能" |
| 问技能建议原因 | "为什么建议这个技能？" |
| 不失主导权的独立评审 | consult reviewer |
| 保存与恢复 | 自动 —— 重开会话说 "continue <task-slug>" |
| 比较两个方案 | fork |
| 跑自动优化 | `/optimize_*` |
| 流水线旧链回退 | 同一命令体，首行改为 `@hm-opt-manager @.opencode/agents/legacy/hm-opt-manager.md` |

**要**：评审走 consult（干净上下文）· 关键决策落到 decisions.md · 把权限拒绝当作
系统在正常工作 · 把好用的组合沉淀成 profile。
**不要**：在角色间粘贴聊天记录（只传胶囊+产物）· 绕过 implementer 的编辑审批 ·
把 draft 当结论引用 · 让 researcher 直接改代码。

## 9. 运行时验证清单（剩余 DoD 项）

迁移的静态门槛已全绿（注册表 lint · pytest · 命令黄金契约）。以下设计要求的检查
需要**真实 OpenCode 会话**，请执行一次并归档结果：

1. **端到端人肉路由任务（M2 DoD）**：assistant → researcher → consult reviewer →
   architect → implementer → reviewer → validator，全程只用胶囊交接。
2. **权限上限（M2/§14）**：让 `@researcher` 改一个源文件 —— 应被运行时拒绝；
   `@reviewer` 同理。确认 implementer 每次编辑都弹审批、设备刷机逐次审批。同时
   探一下 bash 白名单的边界：伪装成只读的变更命令（`find … -delete`、`ls > file`
   重定向）应落入 `ask` 审批而不是放行 —— 若放行，收紧该角色的 bash 模式。
3. **重启恢复（M2 DoD）**：任务中途关闭会话，重开后 "continue <task-slug>" ——
   胶囊必须完整恢复工作状态。
4. **新旧 researcher 深度对比（M2/§14）**：同一目标分别走 `@researcher` 与旧
   `@kernel-research`；领域深度不得下降。
5. **流水线黄金运行 + 对比（M4 DoD）**：同一真实任务分别跑新链 `/optimize_generic`
   与旧链；比较质量/token/轮次并归档；**之后才删除 `agents/legacy/`**（同时清理
   双链映射注记里的旧名字）。
6. **触发词质量（§14）**：用 10 正 + 10 负的任务描述测注册表的
   applies_when/not_for，按误差修订措辞。
7. **Token 度量（§14/§19）**：同一任务分别度量角色会话（仅注册表元数据常驻）与
   迁移前 agent 会话的基础上下文 token，记录差值 —— 设计的渐进披露收益必须量化，
   不能只是断言。
