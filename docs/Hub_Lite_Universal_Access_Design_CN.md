# Hub-Lite：自由会话接入 Memory 与 Skill Hub 的通用接口设计（草案）

> 场景：整个工程开放给全团队后，多数成员不跑优化 pipeline、不用专用 Agent，只是在
> OpenCode / Claude Code 里和 LLM 自由交流解决问题。这些会话里的经验目前无法沉淀，
> 也享受不到团队经验的加持。本设计给出一个"一段配置即接入"的通用形态：读（前向增强）、
> 记（经验捕获）、炼（蒸馏投稿）三个动作全部通过现有 MCP 服务完成，治理线不变。

## 1. 差距分析：为什么自由会话今天沉淀不了

现有沉淀链路建立在三个前提上，自由会话三个都不满足：

| 前提 | pipeline 里怎么满足 | 自由会话的现状 |
|---|---|---|
| 有结构化产物 | 决策台账 `### L00x`、评审否决、bench 报告等固定格式 | 只有对话文本，提炼器认不出 |
| 有收口点 | decision 阶段自动触发提炼钩子 | 聊天没有"结束"信号 |
| 有场景声明 | 每个 run 带 target 函数与阶段 | 用户不会主动写 `mm/vmscan.c::shrink_node` |

消费侧同理：`resolve` 需要 target+stage 才能挂载经验，自由会话没人提供这两个参数。

## 2. 设计原则

1. **一段配置接入**：用户只在自己的 CLAUDE.md（或 opencode 配置）里引一个契约文件 + 注册一个 MCP server，不装任何新东西。
2. **先受益、后贡献**：接入当天就能"问问题前自动带上团队经验"；沉淀是随之而来的副产品，不是额外负担。
3. **治理不让步**：自由会话只是**新的入口**，产物仍然落个人区 → 候选区 → CI 五道门 → 双人审核 → 入库发布，一步不少。易用性绝不换走质量门。
4. **隐私默认本地**：会话产生的任何记录先落个人目录（不进 git），投稿永远是用户显式动作。
5. **静默降级**：hub 不可达时会话照常进行，与现有 MCP 纪律一致。

## 3. 总体架构

```
 任意 LLM 会话（OpenCode / Claude Code，不跑 pipeline）
   │  一次配置：CLAUDE.md 引 hub-lite 契约 + MCP server 注册
   ▼
 hub-lite 行为契约（SKILL.md，托管在 hub、受版本治理）
   │ 何时读 / 何时记 / 何时炼，全部写死在契约里，LLM 照做
   ▼
 Skill Hub MCP 服务（现有 7338 端口，扩 3 个会话级工具）
   ├─ 读  hub_recall(自由文本/符号)  ← 复用混合检索的 free-form 通道
   ├─ 记  hub_log(结构化便签)        → 个人区 chat_journal（gitignored）
   └─ 炼  hub_distill(会话收口)      → 复用规则+大模型双段提炼 → 候选包
                                        │
                     用户过目 → 投稿 PR → CI 五道门 → 双人审核 → 入库 → 版本化发布
                     ▲——————— 与 pipeline 入口汇入同一条治理线，无任何特殊通道 ———————▲
```

关键点：**lite 入口和 pipeline 入口在候选区汇流**，治理层看到的是同一种 schema 合规
候选，只是 `source.kind` 多一种 `chat`。

## 4. 接口设计：在现有 3 个 MCP 工具上扩 3 个会话级工具

### 4.1 `hub_recall(query, k=5)` — 读
- 输入：自由文本（用户的问题原文或 LLM 归纳的一句话），可含符号名。
- 实现：包装现有 `HybridRetriever` 的 free-form 通道；若 query 中能提取出符号/路径
  （LLM 在对话里本来就知道用户聊的是哪个函数），叠加 target-anchored 查询。
- 输出：与 `skillhub_resolve` 同格式的上下文块（事实/教训/已否决方案，带 id 与成熟度），
  lite 模式默认 k=5、约 1.5K token 预算，避免聊天上下文被塞爆。
- 已否决方案照旧标注"不要再提"。

### 4.2 `hub_log(entry)` — 记
- 输入：一条结构化便签 `{type: fact|lesson|bad_plan|idea, claim, target?, mechanism?,
  evidence?, polarity}`。由 LLM 在"结论时刻"（用户确认某方法有效 / 踩了坑 / 纠正了
  一个事实）主动调用，一次一条。
- 落盘：`.opencode/local/chat_journal/<user>/<date>.jsonl`（gitignored，纯个人区），
  返回临时 id 供会话内引用。
- 为什么用"显式便签"而不是全量转录挖掘：噪声低一个量级、天然贴 schema、隐私面小
  （只记结论不记对话）、成本恒定。这与设计文档 §8"两段提取"的取舍一脉相承。

### 4.3 `hub_distill(contributor)` — 炼
- 触发：会话收口时由契约驱动（用户说"今天先到这"、显式 `/distill`、或客户端
  Stop/SessionEnd hook——见 §7 分期）。
- 实现：复用 `sediment` 全链路——journal 便签走新增的确定性映射（便签字段 → schema
  字段几乎一一对应），剩余自由文本走已有的大模型显著性提取；产物照旧进
  `local/sediment_staging/` 并打包。
- 纪律：**绝不自动上传**。返回候选包路径+摘要，用户过目后自行投稿（这也是现有
  sediment 的既定纪律，保持不变）。

`hub_status` 复用现状。三个新工具全部走服务端文件 I/O，任何 MCP 客户端都能调。

## 5. 用户侧配置形态（目标体验）

CLAUDE.md（个人或项目级）加一行：

```markdown
@hub/skills/hub-lite/SKILL.md
```

MCP 注册（以 Claude Code 为例，OpenCode 等价）：

```json
{ "mcpServers": { "skillhub": { "url": "http://<平台>:7338/mcp" } } }
```

加上环境变量 `HMOPT_HUB_USER=<姓名>`（投稿署名用）。三步完成，此后：

- 用户问"shrink_node 为什么这么慢"→ LLM 按契约先 `hub_recall`，回答自动引用
  F001/H001，且不会再提 B001 那类已否决方案；
- 会话中用户确认"这招管用/这是个坑"→ LLM 悄悄 `hub_log` 一条便签；
- 收口时 `hub_distill`，用户看一眼候选包决定要不要投稿；
- 即使从不投稿，journal 也会进入**本地检索叠加层**（resolver 的 local overlay 机制
  现成支持）——自己的经验下次自己先用上。

## 6. hub-lite 行为契约（SKILL.md 草案要点）

契约本身托管在 hub 的 `skills/` 下——**它自己就是一个受治理、可进化的技能**：
版本化发布、可被评测门优化，用户引用的是发布版而非漂移的草稿。核心条款：

1. 会话涉及内核/性能/优化话题时，回答前先 `hub_recall`；命中记录须引用 id；
   命中已否决方案的思路不得再提议。
2. 出现"结论时刻"（方法被验证有效/无效、事实被纠正、坑被踩实）即 `hub_log`
   一条便签；宁缺毋滥，一次会话通常 0–5 条。
3. 会话收口或用户要求时调用 `hub_distill`，向用户播报候选摘要与投稿命令。
4. hub 不可达时静默跳过全部三个动作，不打扰会话。
5. 敏感信息（序列号、密钥、内部主机名）不得写入便签——脱敏门会拦，但第一道
   责任在契约。

## 7. 实现拆解与分期（大量复用，新增代码很少）

| 期 | 内容 | 改动点 |
|---|---|---|
| **P0（约一周）** | 3 个 MCP 工具 + hub-lite 契约 + journal 提炼映射 | `skillhub_mcp_service.py` 加 handler（recall 是 resolver 包装）；`sediment` 加 chat_journal reader；schema `source.kind` 增 `chat`（小版本）；新增 `skills/core/hub-lite/SKILL.md`；配置样例文档 |
| **P1** | 自动收口 + 本地叠加检索 | Claude Code hooks / OpenCode plugin 在会话结束自动 `hub_distill`；journal 纳入 resolver 的 local overlay 源 |
| **P2** | 个性化消费 | 按 user 画像/历史命中重排序检索结果（与 slide 创新点 3"场景化智能消费"的个性化推荐收口） |

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| 聊天经验噪声大、质量参差 | chat 来源默认 tentative/L1；确认数阈值与七类判定照常把关；门控审核一步不少 |
| 隐私泄露面变大 | journal 个人区不进 git；投稿必经人工过目；脱敏扫描对 chat 候选强制执行 |
| 检索注入污染聊天上下文 | lite 模式收紧 k 与 token 预算（5 条 / 1.5K），沿用分阶段预算思想 |
| 契约漂移（各人魔改 prompt） | 契约托管在 hub、版本化发布，用户引用发布版；魔改者得不到升级 |
| 便签滥记刷量 | 投稿仍按 contributor 分目录、审核可见人；晋升需独立复现，刷不动 |

## 9. 对整体方案叙事的增益

这一层补上后，方案定位从"优化 pipeline 的配套记忆"升级为"**任何 LLM 会话皆可
接入的团队经验基座**"——slide 上"通用：任意 Agent 优化场景即插即用"和创新点 3
"场景化智能消费"从愿景变成了带接口定义的落地路径；对评审的说法也顺了：
专家跑 pipeline 产出高质量经验，全员在日常会话中消费并回馈长尾经验，
两路汇入同一条治理线——覆盖面和经验密度同时上去，飞轮转得更快。
