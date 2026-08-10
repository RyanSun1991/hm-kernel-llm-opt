# HMOPT 平台总览与快速上手(Multi-Agent + 记忆 + 索引)

面向:所有团队成员(第一次接触本平台的人从这里开始)
目标:看懂整体架构 → 30 分钟部署起来 → 5 分钟开始用
深入阅读:每节末尾给了对应的详细文档。

---

## 1. 这套系统是什么(三句话)

1. 一个围绕 **OpenCode** 的**多角色 AI 工作台**:你和 7 个通用角色(研究/设计/实现/
   评审/验证…)直接对话解决内核问题,**你决定下一步给谁**;也可以一条命令跑全自动优化
   pipeline。
2. 一套**团队记忆系统**:对话中的经验(踩坑、结论、方法)被随手记下,经确认沉淀进
   团队知识库(Skill Hub),下次任何人提问都能自动召回。
3. 一组**索引与工具服务**(MCP):内核代码索引、构建、真机测试、烧写 —— agent 干活
   时调用的"手和眼睛"。

## 2. 一张图看懂整体架构

```
                你(团队成员,OpenCode 里对话)
                          │
        ┌─────────────────┴──────────────────┐
        │      工作台(.opencode/)            │
        │  7 个通用角色:assistant(默认入口)  │
        │  researcher·architect·implementer  │
        │  reviewer·validator·coordinator    │
        │  + 技能库(按需加载,注册表索引)     │
        │  + 任务工作区(task/capsule/工件)   │
        └───┬──────────────┬──────────────┬──┘
            │调用           │记忆           │pipeline 模式
            ▼              ▼              ▼
   ┌────────────────┐ ┌──────────────┐ ┌─────────────────┐
   │ MCP 工具服务    │ │ 记忆三层      │ │ /optimize_* 命令 │
   │ 代码索引 7332   │ │ journal(个人)│ │ coordinator 驱动 │
   │ git 7334       │ │   ↓ 沉淀(确认)│ │ 同一批角色走     │
   │ 构建 7335      │ │ hub 知识(团队)│ │ 强制阶段门       │
   │ 真机测试 7336   │ │   ↓ 策展(评审)│ └─────────────────┘
   │ 烧写 7337      │ │ hub 技能(方法)│
   │ skill-hub 7338 │ └──────────────┘
   └───────┬────────┘
           ▼
   内核仓库(hm-verif-kernel)+ Windows 中继 + 真机设备
```

一句话流程:**你提问 → 角色召回团队经验 + 查代码索引 → 干活产出工件 → 有价值的经验
记入个人 journal → 你确认后沉淀进团队 hub → 反哺所有人的下一次提问。**

## 3. 三大块分别是什么

### 3.1 Multi-Agent 工作台(默认交互方式)

**解决的问题**:以前是全自动 pipeline,用户插不上手;角色和优化目标焊死,非优化任务
不能用。现在角色通用、你主导。

要点:

- **默认和 `assistant` 说话**,简单问题直接答;复杂任务它建议"找 researcher 开个
  任务",**你确认才走**;
- **7 个角色各管一段**:researcher 查清事实(改不了代码,权限硬限制)→ architect 出
  方案 → implementer 改代码(每次编辑要你批准)→ reviewer 独立评审 → validator 真机
  验证;每步结束角色给出"下一步建议 + 可直接转发的任务简报",**转不转你说了算**;
- **技能按需加载**:领域知识(reclaim/hyperhold/优化方法)都在技能包里,角色开工时
  按任务匹配、建议 ≤3 个、你确认才加载 —— 上下文不臃肿;
- **profile 一步到位**:`@reclaim-investigator` 等 6 个预组合,等于"角色+领域包"
  一键进入;
- **任务工作区**:每个任务一个目录(目标/进展摘要/工件/决策),换角色、断线重连都
  不丢现场;
- **老 pipeline 还在**:`/optimize_workqueue` 等命令照常,coordinator 自动驱动全流程。

> 详细:`docs/Agent_Workbench_Usage_CN.md`(使用)· `docs/Agent_Workbench_Design_CN.md`(设计)

### 3.2 记忆系统(经验怎么流动)

**解决的问题**:一个人踩过的坑,换个人重踩;好方法散在聊天记录里找不回来。

三层结构,逐层升级:

```
journal(个人,未审)──沉淀(你确认)──▶ hub 知识(团队,已策展)──评测──▶ hub 技能(方法论)
```

要点:

- **自动记**:对话中出现 6 类有价值信号(验证过的结论/你的裁决/结构性事实/踩坑根因/
  好用配方/纠正旧知识)时,LLM 自动调 `memory_log` 记一条到你的个人 journal;拿不准
  会问你"记一下吗?";你说"**记一下**"则必记;
- **自动查**:角色提方案前先 `memory_recall`,返回结果分层标注(`journal·未审` /
  `hub·已策展`),引用带 ID;
- **进团队库要过闸**:会话收尾你确认后 `skillhub_sediment` → hub staging → PR →
  5 道 CI 门 → 人工策展 → 正式知识(F/H/A/B 编号)。**没有任何内容会不经你确认进入
  团队库**;
- **不跑 pipeline 的成员也能用**:任何仓库里配一个 MCP 地址 + CLAUDE.md 贴一段,
  两步接入(见 §5.3)。

> 详细:`docs/Team_Memory_Onboarding_CN.md`(接入)· `docs/Team_Memory_Design_CN.md`(设计)· `docs/Skill_Hub_Runbook_CN.md`(hub 运维)

### 3.3 索引与工具服务(agent 的手和眼睛)

**解决的问题**:agent 空口分析内核不可靠,要能查真代码、跑真构建、上真机。

| 服务 | 端口 | 干什么 |
|---|---|---|
| 代码索引 MCP | 7332(容器内 7331) | clangd + 向量索引:查符号、调用链、热点上下文 |
| Sequential Thinking | 7333 | 分步推理辅助 |
| Git MCP | 7334 | 仓库操作 |
| 构建 MCP | 7335 | 触发内核构建/签名 |
| 真机测试 MCP | 7336 | 指令数测试、**lmbench 全套 A/B**(经 Windows 中继) |
| 烧写 MCP | 7337 | 刷 stock/feature 镜像(经 Windows 中继) |
| Skill-Hub MCP | 7338 | 记忆三件套(log/recall/sediment)+ hub 读写 |
| REST API | 8001 | /runs、/metrics、/report 查询 |

> 详细:`docs/Kernel_Index_MCP_Onboarding_zh.md` · `docs/OpenCode_MCP_Integration_Guide.md`

## 4. 快速部署(约 30 分钟,一台 Linux 服务器)

前置:Docker + docker-compose;一个可用的 LLM 网关地址;内核仓库 checkout。

```bash
# 1) 取代码
git clone <本仓库> && cd hm-kernel-llm-opt && git checkout opencode

# 2) 配环境(只有 3 个必填)
cp .env.example .env
#   HMOPT_LLM_BASE_URL=http://<你的LLM网关>:<port>/v1     ← 必填
#   HMOPT_LLM_API_KEY=<key>                               ← 必填
#   KERNEL_REPO_PATH=/path/to/hm-verif-kernel             ← 必填
#   (真机链路才需要:HMOPT_FLASH_RELAY_URL=http://<Windows中继>:9100 及镜像目录)

# 3) 起服务
docker compose up -d
docker compose ps          # 应看到 hmopt / git-mcp / build-mcp / skillhub-mcp 都 Up

# 4) 建代码索引(首次,较久;之后增量)
docker exec -it hmopt hmopt index-kernel \
  --repo-path /workspace/kernel --compile-commands-dir /workspace/kernel

# 5) 验证
curl -s localhost:8001/health          # REST 存活
curl -s localhost:7338/mcp -o /dev/null -w "%{http_code}\n"   # skill-hub MCP 可达
docker exec -it hmopt hmopt query "who calls shrink_node" --mode code   # 索引可查
```

常见问题:LLM 网关不通 → 查 `.env` 的 BASE_URL 从容器内可达(`host.docker.internal`
或内网 IP);索引查询报 Docstore empty → 第 4 步没跑完。

## 5. 快速上手(5 分钟)

### 5.1 第一次会话(工作台模式,推荐)

在内核仓库目录打开 OpenCode,直接说话(默认就是 assistant):

```
你:shrink_node 里 sc->priority 是干嘛的?
assistant:(直接解释,给 file:line)         ← 简单问题,到此为止

你:帮我彻底查一下 shrink_node 的竞态嫌疑
assistant:这值得开个任务,建议交给 researcher,简报我拟好了,发吗?
你:发
researcher:已开工作区。建议加载 domain-reclaim + method-lifetime 两个技能(理由…),确认?
你:好
researcher:(查索引、读代码……)结论:… (evidence: vmscan.c:137-155)
           下一步建议:1. consult reviewer 挑战这个分析  2. handoff architect 出方案
你:1                                        ← 你永远是路由器
```

过程中看到有价值的结论,顺口说 **"记一下"** → 进你的 journal;会话结束 LLM 会盘点
"本次有 2 条可沉淀,确认吗?" → 确认后进入团队库流程。

### 5.2 跑全自动优化 pipeline(老用法,没变)

```
/optimize_workqueue      # coordinator 自动驱动:研究→方案评审→实现→代码评审→真机A/B
```

### 5.3 新成员接入记忆系统(不需要 clone 本仓库,两步)

任意自己的仓库里:

```jsonc
// ① .mcp.json / opencode.json 加一行
{ "mcpServers": { "skill-hub": { "type": "http", "url": "http://<服务器>:7338/mcp" } } }
```

```markdown
② 自己的 CLAUDE.md / AGENTS.md 贴接入片段(完整版见 Team_Memory_Onboarding_CN.md):
   要点 = 提案前先 recall;出现 6 类信号就 memory_log;收尾确认后 sediment;密钥不入库
```

之后你的日常对话就自动进入"查得到团队经验、记得下个人经验"的循环。

### 5.4 速查表

| 想做什么 | 怎么做 |
|---|---|
| 换角色 | Tab 或 `@researcher`,或采纳上一轮的"下一步建议" |
| 一键进入领域组合 | `@reclaim-investigator` `@hyperhold-io` `@workqueue` `@sync-mechanism` `@kernel-understand` `@bug-fix` |
| 加/减技能 | "加载 memory-tlb" / "卸掉 IC 技能" |
| 记一条经验 | 说"记一下" |
| 查团队经验 | 角色会自动 recall;也可直接问"查一下 hub 里关于 X 的经验" |
| 沉淀进团队库 | 会话收尾按提示确认(或说"沉淀一下") |
| 恢复上次任务 | "继续 <任务名>"(工作区自动持久) |
| 全自动优化 | `/optimize_generic|workqueue|hyperhold|memmgr_reclaim` |

## 6. 再深入一层(按需阅读)

| 主题 | 文档 |
|---|---|
| 工作台完整用法(场景示例、fork、评审) | `Agent_Workbench_Usage_CN.md` |
| 工作台设计原理(角色/技能/权限/工件状态) | `Agent_Workbench_Design_CN.md` |
| 记忆系统接入与条目格式 | `Team_Memory_Onboarding_CN.md` |
| Skill Hub 策展与发版运维 | `Skill_Hub_Runbook_CN.md` |
| pipeline 车道规范(阶段门/交接) | `.opencode/docs/harness_engineer_system.md` |
| lmbench 真机 A/B 协议 | `.opencode/skills/scenario/kernel-opt/ab-test-comparison-lmbench/SKILL.md` |
