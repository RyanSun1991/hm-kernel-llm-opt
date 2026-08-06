# HM-VERIF Pipeline Demo Video — 拍摄脚本（Shooting Script）

> 配套文档：`docs/Pipeline_Demo_Video_Plan.md`（总体方案）
> 时长基线：**8 分钟 · 1080p30 · 无旁白 · 中文字幕卡 + 屏幕注释**
> 结构：用户输入 → 每一步执行 → harness 多智能体协作 → 结果与沉淀 → 闭环迭代

---

## 全局说明

**镜头记法**：每个 SCENE 内部按 BEAT 拆，每个 BEAT 标 5 列：
`画面 / 终端 or UI 文本 / 字幕卡 / 屏幕注释 / 音效`。
"终端文本"列里给的内容是**镜头预期形态的示意稿**，正式录制时以真实跑批输出为准——脚本只规定"哪些行必须留在镜头里、必须被注释圈出"。

**HUD 元件（贯穿全片）**：屏幕右上角始终挂一枚 96×96 px 的微型架构 HUD——hub-and-spoke 图，当前激活的 agent 节点亮蓝色。它是把 ACT 3"多智能体协作"持续可视化的关键手段，不需要额外停顿讲解。

**主屏分区命名**（与 plan §5.3 一致）：左上 = OPENCODE 区，右上 = MCP 区，左下 = FS/DIFF 区，右下 = DASH 区。"PiP"指事后从独立录的子源（scrcpy / 单独 MCP 终端）裁出来贴上去的子画面。

---

## ACT 1 — 用户输入（0:00 – 0:30）

### SCENE 1.1 — Cold Open（0:00 – 0:14）

| BEAT | t | 画面 | 终端/UI 文本 | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|---|
| 1 | 0:00–0:03 | 纯黑 → 大字号数字依次砸入屏幕中央 | — | `7 STAGES · 4 DOMAINS · 2 GATES · 1 CLOSED LOOP` | — | 低频 bass drop |
| 2 | 0:03–0:09 | 4 分屏 0.6s 闪现：OpenCode 终端、MCP 多窗格、手机 scrcpy、指标看板 | — | — | 四角小标签：`OPENCODE` `MCP` `DEVICE` `METRICS` | BGM 起 |
| 3 | 0:09–0:14 | 平台 Logo 中央升起 + 主副标题 | — | 主：`HM-VERIF Kernel LLM 优化平台`<br>副：`闭环 Pipeline 自动迭代演示` | — | sustain |

### SCENE 1.2 — 用户输入（0:14 – 0:30）

| BEAT | t | 画面 | 终端/UI 文本 | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|---|
| 1 | 0:14–0:18 | OPENCODE 区全屏放大；光标在空 prompt 闪烁 | `user@dev-host:~/hm-kernel-llm-opt$ █` | — | — | 环境氛围音 |
| 2 | 0:18–0:25 | 用户一字一字敲入命令（按真实击键节奏） | `bash scripts/run_opencode_pipeline.sh \`<br>`  --profile generic_full \`<br>`  --target sysmgr/memmgr/...{TARGET}.c \`<br>`  --start-mcp` | — | 黄框圈住 `--profile generic_full` 与 `--target` | 键盘 ASMR |
| 3 | 0:25–0:30 | 回车，刷出第一段绿色 OK 行 | `[start] main MCP via scripts/run_mcp_server.sh`<br>`[start] build MCP via scripts/run_build_mcp_server.sh`<br>`[start] auto-test MCP via scripts/run_auto_test_mcp_server.sh`<br>`[start] seq MCP via scripts/run_seq_mcp_server.sh`<br>`[ok] staged prompt → .opencode/state/current_prompt.md` | — | 浅蓝标签贴在最后一行：`pipeline session staged` | "ping" |

> **录制要点**：本场是全片唯一"有人介入"的画面。整段之后到 ACT 5 末尾不出现任何鼠标光标或人手。

---

## ACT 2 — 每一步执行（0:30 – 5:50）

> ACT 2 的 6 个 SCENE 共用以下规则：
> - 每个 SCENE 开头 5s 章节字幕卡（黑底）
> - HUD 在 SCENE 切换瞬间把当前节点切到对应 agent
> - 文件落盘瞬间用黄色脉冲框 + "ping" 音效强提示

### SCENE 2.1 — Stage 1 · Intake & Routing（0:30 – 1:05）

| BEAT | t | 画面 | 终端/UI 文本 | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|---|
| 1 | 0:30–0:35 | 字幕卡全屏 | — | 主：`Stage 1 — Intake & Routing`<br>副：`入站与路由`<br>角：`01 / 07` | — | stinger |
| 2 | 0:35–0:43 | OPENCODE 区出现首个 banner | `=== kernel-pipeline-starter v1 — acknowledging target: {TARGET} ===`<br>`loading: .opencode/config.yaml`<br>`loading: .opencode/skills/infra/language-config.md`<br>`loading: .opencode/docs/harness_engineer_system.md`<br>... | — | 浅蓝标签：`agent: kernel-pipeline-starter v1` | — |
| 3 | 0:43–0:52 | 自动切到 manager；HUD 中心 hub 亮起 | `=== os-opt-manager v1 — taking control of target: {TARGET} ===`<br>`stage: intake`<br>`primary metric: instruction_count`<br>`routing decision: specialist = auto (generic_full)` | — | 浅蓝标签：`agent: os-opt-manager v1`<br>HUD 中心节点：`MANAGER` 亮 | — |
| 4 | 0:52–1:05 | manager 输出首个 handoff packet（右侧弹浮窗） | `## Handoff → research`<br>`- target: {TARGET}`<br>`- primary_metric: instruction_count`<br>`- evidence_baseline: outputs/runs/{run_id}/baseline.json`<br>`- hot_path: (to be discovered)`<br>`- files_in_scope: [{TARGET}]`<br>`- risks: routing-only stage, no code change`<br>`- next_action: research specialist enter` | — | 右侧浮窗 5 字段逐项绿色脉冲 | tick × 5 |

### SCENE 2.2 — Stage 2 · Research（1:05 – 1:55）

| BEAT | t | 画面 | 终端/UI 文本 | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|---|
| 1 | 1:05–1:10 | 字幕卡 | — | 主：`Stage 2 — Research`<br>副：`调研`<br>角：`02 / 07` | — | stinger |
| 2 | 1:10–1:25 | OPENCODE 区 specialist banner + 调用 MCP；MCP 区高亮 | OPENCODE：`=== {specialist} v1 — acknowledging target ===`<br>`tool_call: mcp__hmopt__query("call chain of {FN}", mode="runtime_code")`<br>`tool_call: mcp__hmopt__symbol_neighbors("{FN}")`<br><br>MCP：`[main]  ▶ query(mode=runtime_code) hits=27 latency=412ms`<br>`[main]  ▶ symbol_neighbors hits=14 latency=98ms` | — | OPENCODE 一侧黄框圈 `tool_call:` 行<br>MCP 一侧黄框圈对应响应行<br>HUD：`RESEARCH` 节点亮 | 数据流 whoosh |
| 3 | 1:25–1:40 | FS/DIFF 区切到文件树；两个新文件淡入 | 文件树左侧高亮：<br>`.opencode/docs/{target}_design.md`（新增）<br>`.opencode/plans/{target}_plan.md`（新增） | — | 双黄框，每个框边写中文：`design 笔记` / `优化候选 plan` | "ping" × 2 |
| 4 | 1:40–1:55 | research 日志 ×4 加速扫过；关键命中行减速回 1× | （正常滚动）...<br>`(slow) HOT-SPOT: {FN} contributes 18.4% of cycles`<br>`(slow) IDEA #1: hoist boundary check out of inner loop`<br>`(slow) IDEA #2: replace linear scan with hashed lookup`<br>`(slow) IDEA #3: ...` | — | 屏角字幕：`× 4 加速`（仅在加速段显示） | tape rewind 拟态 |

### SCENE 2.3 — Stage 3 · Plan Review（GATE）（1:55 – 2:35）

| BEAT | t | 画面 | 终端/UI 文本 | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|---|
| 1 | 1:55–2:00 | 字幕卡（GATE 专用样式：底色泛红警示边） | — | 主：`Stage 3 — Plan Review (GATE)`<br>副：`计划评审 — 不通过即回退`<br>角：`03 / 07` | — | gate-stinger（低频） |
| 2 | 2:00–2:15 | OPENCODE 区 reviewer banner；逐条评分输出 | `=== kernel-plan-reviewer v1 — reviewing plan ===`<br>`criterion: instruction-count tradeoff ........... PASS`<br>`criterion: risk surface (locks / refcount) ...... PASS`<br>`criterion: testability (A/B feasibility) ........ PASS`<br>`criterion: bad-plan memory check ................ PASS` | — | 浅蓝标签：`agent: kernel-plan-reviewer v1`<br>右侧侧栏中文："这是硬闸门：未通过则回退到 Research" | — |
| 3 | 2:15–2:28 | FS/DIFF 区切到 `*_plan_review.md`；verdict 行放大 | `verdict: APPROVED`<br>`top_idea: IDEA #1 — hoist boundary check`<br>`proceed_to: implementation` | — | 绿色对勾 ✓ 动画从 verdict 行放大覆盖 | gate-pass chime（高频 ding） |
| 4 | 2:28–2:35 | OPENCODE 区 manager 自动 dispatch | `[manager] received verdict: APPROVED`<br>`[manager] dispatching → kernel-code-agent` | — | 屏底字幕条："✓ 通过，pipeline 自动进入实现阶段"<br>HUD 节点流向 `IMPL` | — |

### SCENE 2.4 — Stage 4 · Implementation（2:35 – 3:30）

| BEAT | t | 画面 | 终端/UI 文本 | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|---|
| 1 | 2:35–2:40 | 字幕卡 | — | 主：`Stage 4 — Implementation`<br>副：`实现`<br>角：`04 / 07` | — | stinger |
| 2 | 2:40–2:55 | OPENCODE 一半 + FS/DIFF 一半同屏；diff 实时滚动 | OPENCODE：`=== kernel-code-agent v1 — implementing IDEA #1 ===`<br>`editing: {TARGET}`<br>`tool_call: mcp__build__compile(target={MODULE})`<br><br>FS/DIFF：`+ if (likely(!boundary_dirty(ctx)))`<br>`+   return fast_path(ctx);`<br>`-` ... | — | 浅蓝标签：`agent: kernel-code-agent v1`<br>FS 一侧绿/红 diff 行用 +/- 色块自然显示 | 键盘 ASMR |
| 3 | 2:55–3:15 | MCP 区切到 build MCP；进度条 ×8 加速 | `[build] ▶ make ARCH=arm64 CC=clang ...`<br>`[build]   CC kernel/{X}.o`<br>`[build]   CC kernel/{Y}.o`<br>`...（加速段）...`<br>`[build] ✓ BUILD PASS  duration=7m12s` | — | 屏角字幕：`× 8 加速 · build 实际耗时 7m 12s`<br>build 终端外圈 orange hue 边框 | 加速 hum |
| 4 | 3:15–3:30 | FS/DIFF 区：patch 文件 + `git diff --stat` 弹出 | `.opencode/patches/{target}.patch  (新增)`<br><br>`$ git diff --stat`<br>` {TARGET}  │ 17 +++++++++++-----`<br>` 1 file changed, 12 insertions(+), 5 deletions(-)` | — | 黄框圈 patch 文件名<br>绿色大字 `BUILD PASS` 浮起 | success chime |

### SCENE 2.5 — Stage 5 · Code Review（GATE）（3:30 – 4:10）

| BEAT | t | 画面 | 终端/UI 文本 | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|---|
| 1 | 3:30–3:35 | 字幕卡（GATE 警示边） | — | 主：`Stage 5 — Code Review (GATE)`<br>副：`代码评审 — 不通过即回退`<br>角：`05 / 07` | — | gate-stinger |
| 2 | 3:35–3:55 | OPENCODE 区 reviewer 输出；tradeoff 段落放大 | `=== kernel-code-reviewer v1 — reviewing patch ===`<br>`section: correctness ..................... PASS`<br>`section: instruction-count tradeoff ...... +Δ -12.3% est`<br>`section: locking & refcount .............. PASS`<br>`section: build & symbol shape ............ PASS` | — | 框出 `tradeoff` 段；右侧浮窗："Reviewer 估算指令数 ↓ 12.3%（待真机验证）" | — |
| 3 | 3:55–4:05 | FS/DIFF 区切到 `*_code_review.md`；verdict 放大 | `verdict: APPROVED`<br>`unblocked: tester`<br>`notes: see tradeoff section for follow-up` | — | 绿色对勾 ✓ | gate-pass chime |
| 4 | 4:05–4:10 | OPENCODE 区 manager 自动 dispatch | `[manager] received verdict: APPROVED`<br>`[manager] dispatching → kernel-tester-agent` | — | 屏底字幕条："✓ 通过，进入真机 A/B 验证"<br>HUD 节点流向 `TEST` | — |

### SCENE 2.6 — Stage 6 · Tester A/B（4:10 – 5:50）★ 重头戏

> 本场强制保留四分屏布局始终可见；以下 BEAT 描述"哪一区是焦点"。

| BEAT | t | 焦点 | 终端/UI 文本 | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|---|
| 1 | 4:10–4:15 | 字幕卡 | — | 主：`Stage 6 — Tester A/B Validation`<br>副：`真机 A/B 验证`<br>角：`06 / 07` | — | stinger |
| 2 | 4:15–4:25 | OPENCODE | `=== kernel-tester-agent v1 — A/B plan ===`<br>`step 1/4: flash stock`<br>`step 2/4: run workload on stock`<br>`step 3/4: flash feature`<br>`step 4/4: run workload on feature` | — | 浅蓝标签<br>右侧浮窗：A/B 4 步流程 checklist | — |
| 3 | 4:25–4:40 | MCP（flash）+ DEVICE | MCP：`[auto-test] ▶ hdc shell reboot bootloader`<br>`[flash]     ▶ flash boot stock.img`<br>`[flash]     ▶ flash system stock.img`<br>`[flash]     ▶ reboot`<br><br>scrcpy：fastboot 进度条 → 解锁桌面 | — | scrcpy 一侧贴红字：`Stock Image`<br>MCP 一侧 magenta hue 边框（auto-test）<br>HUD 节点：`TEST` 持续亮 | flash 进度音 |
| 4 | 4:40–4:55 | DEVICE + DASH | DEVICE：workload 应用启动并跑测<br>DASH（4 路指标 baseline 数字定格）：<br>`instruction_count = 1,234,567`<br>`cycles            =   980,221`<br>`cache_miss        =    51,388`<br>`wakeup_lat (us)   =       144` | — | DASH 第一列上方贴红字：`Baseline`<br>每条指标定格瞬间 tick 音 | tick × 4 |
| 5 | 4:55–5:00 | 全屏中插字幕卡 | — | `切换镜像中...` | — | swoosh |
| 6 | 5:00–5:15 | MCP（flash）+ DEVICE | MCP：`[flash] ▶ flash boot feature.img`<br>`[flash] ▶ flash system feature.img`<br>`[flash] ▶ reboot`<br><br>scrcpy：fastboot → 桌面（同节奏） | — | scrcpy 一侧贴绿字：`Feature Image` | flash 进度音 |
| 7 | 5:15–5:30 | DEVICE + DASH | DEVICE：workload 第二次启动<br>DASH（第二列定格）：<br>`instruction_count = 1,083,219`<br>`cycles            =   866,750`<br>`cache_miss        =    47,901`<br>`wakeup_lat (us)   =       138` | — | DASH 第二列上方贴绿字：`Feature`<br>tick 音节奏与 Baseline 一致 | tick × 4 |
| 8 | 5:30–5:42 | DASH 全屏放大 | DASH 第三列"Δ"出现，数字从 Baseline 数到 Feature：<br>`Δ instruction_count  ↓ 12.3%`<br>`Δ cycles             ↓ 11.6%`<br>`Δ cache_miss         ↓  6.8%`<br>`Δ wakeup_lat         ↓  4.2%` | — | 主指标行 96pt 大字 + 红色 ↓ 箭头；其它指标 48pt | drop（主指标定格瞬间） |
| 9 | 5:42–5:50 | FS/DIFF | `.opencode/bench/{target}_validation.md` 弹出，第一行高亮：<br>`verdict: IMPROVED — primary metric ↓ 12.3%` | — | 黄框圈文件名；绿色对勾 ✓ | "ping" + chime |

---

## ACT 3 — Harness 多智能体协作（贯穿 + 集中展示）

ACT 3 不占独立时间段，分两层呈现：

**贯穿层**：HUD 全片在线，每次 agent 切换都把对应节点亮起；这就是 ACT 2 各场上 `HUD 节点：xxx 亮` 注释的来源。

**集中展示层**（4:05–4:25 之间穿插 12s "macro view"，叠在 SCENE 2.6 BEAT 1–2 之上）：

| BEAT | t | 画面 | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|
| M1 | 4:05–4:09 | 镜头从右上 HUD "拉近"到全屏 — 中心 `os-opt-manager` 节点，向外辐射 6 个 specialist | — | 节点旁标注：`research / plan-reviewer / coder / code-reviewer / tester / memory` | swell |
| M2 | 4:09–4:17 | 7 条曾经触发过的 handoff 连线按时间顺序点亮（与本片实际触发顺序一致），每条线尾端弹出 artifact 缩略图（plan.md / review.md / patch / validation.md） | — | 屏底字幕条：`Hub-and-Spoke · 仅 manager 持 delegate 权 · 所有 sub-agent 完成后回 manager` | tick × 7 |
| M3 | 4:17–4:25 | 镜头拉回 HUD 角落；4 分屏画面浮回 | — | 屏底字幕条淡出 | settle |

> 这段是把"为什么这是真闭环、不是流水线脚本"讲清楚的关键 12 秒。所有动画素材在 plan §11 D-11 那天预先做好；正片只是把动画 overlay 到这段录屏之上。

---

## ACT 4 — 结果与沉淀（5:50 – 6:30）

### SCENE 4.1 — Stage 7 · Decision & Memory（5:50 – 6:15）

| BEAT | t | 画面 | 终端/UI 文本 | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|---|
| 1 | 5:50–5:55 | 字幕卡 | — | 主：`Stage 7 — Decision & Memory`<br>副：`决策与长期记忆`<br>角：`07 / 07` | — | stinger |
| 2 | 5:55–6:05 | OPENCODE 区 manager 落定决策 | `[manager] reading: .opencode/bench/{target}_validation.md`<br>`[manager] verdict: ACCEPT — instruction_count ↓ 12.3%`<br>`[manager] writing memory: .opencode/memory/targets/{target}.md`<br>`[manager] writing memory: .opencode/memory/ideas/accepted.md` | — | 浅蓝标签：`agent: os-opt-manager v1`<br>HUD 节点流向 `MEMORY` | — |
| 3 | 6:05–6:15 | FS/DIFF 区文件树滚动；本轮新增/修改文件依次绿色 ↑ 闪 | （文件树列高亮以下条目，每条 0.4s）<br>`.opencode/docs/{target}_design.md            (+)`<br>`.opencode/plans/{target}_plan.md             (+)`<br>`.opencode/reviews/{target}_plan_review.md    (+)`<br>`.opencode/patches/{target}.patch             (+)`<br>`.opencode/reviews/{target}_code_review.md    (+)`<br>`.opencode/bench/{target}_validation.md       (+)`<br>`.opencode/memory/targets/{target}.md         (±)`<br>`.opencode/memory/ideas/accepted.md           (±)`<br>`outputs/runs/{run_id}/metrics.json           (+)` | — | 每条文件名右侧贴中文小标：`research笔记` / `候选 plan` / `计划评审` / `代码补丁` / `代码评审` / `A/B 验证报告` / `目标长期记忆` / `已采纳想法库` / `指标快照` | "ping" × 9，节奏渐密 |

### SCENE 4.2 — 本轮 artifacts 总览（6:15 – 6:30）

| BEAT | t | 画面 | 终端/UI 文本 | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|---|
| 1 | 6:15–6:30 | 全屏汇总卡：3 列分别列 plans/reviews/patches/bench/memory 的所有路径；中间一行大字总计 | （静态卡） | 卡顶大字：`9 份可审计 artifacts`<br>卡底小字：`零人工介入 · 全部按约定路径落盘` | 卡中央 KPI："本轮新增 N 个文件 · 修改 M 个文件 · 总耗时 T 分钟" | sustain |

---

## ACT 5 — 闭环迭代（6:30 – 8:00）

### SCENE 5.1 — Round 2 自动启动（6:30 – 6:55）

| BEAT | t | 画面 | 终端/UI 文本 | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|---|
| 1 | 6:30–6:35 | 字幕卡（节奏更急促） | — | 主：`Iteration #2 — Closed-Loop Re-Entry`<br>副：`第 2 轮迭代自动启动 · 无人介入` | — | stinger（高昂） |
| 2 | 6:35–6:50 | OPENCODE 区 manager 再次 fan-out | `[manager] iteration #2 — re-entering pipeline`<br>`[manager] reading prior memory: .opencode/memory/targets/{target}.md`<br>`[manager] re-routing → specialist: {same or different}`<br>`[manager] dispatching → research` | — | HUD 节点回到 `RESEARCH` 并亮起 | — |
| 3 | 6:50–6:55 | 4 分屏快剪 0.5s 闪现 4 个画面（OpenCode / MCP / DEVICE / DASH） | — | — | — | drum fill |

### SCENE 5.2 — Round 2 极速快进（6:55 – 7:25）

> 整段 30 秒 ×16 倍速覆盖第 2 轮的全部 7 个 stage，**仅保留 banner 行与 verdict 行**作为关键帧。

| BEAT | t | 画面 | 终端文本（仅保留行） | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|---|
| 1 | 6:55–7:25 | OPENCODE 区主，HUD 节点依次亮：RESEARCH → PLAN-REVIEW → IMPL → CODE-REVIEW → TEST → MEMORY | `=== {specialist} v1 — iteration 2 ===`<br>`...`<br>`verdict: APPROVED`<br>`=== kernel-code-agent v1 — iteration 2 ===`<br>`...`<br>`verdict: APPROVED`<br>`=== kernel-tester-agent v1 — iteration 2 ===`<br>`verdict: IMPROVED — primary metric ↓ 4.7% (incremental)` | — | 屏角字幕：`× 16 加速 · 第 2 轮实际耗时 25m`<br>HUD 节点每次切换一次 tick 音 | 持续加速 hum + 7 × tick |

### SCENE 5.3 — 累计指标对比（7:25 – 7:45）

| BEAT | t | 画面 | 终端/UI 文本 | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|---|
| 1 | 7:25–7:45 | DASH 全屏；双柱对比图（Baseline / Iter 1 / Iter 2）从左到右生成 | `Baseline      Iter 1        Iter 2`<br>`1,234,567 →  1,083,219 →   1,032,440`<br>`           ↓ 12.3%       ↓ 16.4% 累计`<br>`           ↓ 4.7% 增量` | — | 主指标 96pt 大字：`累计指令数 ↓ 16.4%`<br>第 2 轮增量绿色脉冲<br>屏底小字："累计两轮闭环 · 零人工代码改动" | drop（双柱填满时） |

### SCENE 5.4 — Outro（7:45 – 8:00）

| BEAT | t | 画面 | 文本 | 字幕卡 | 屏幕注释 | 音效 |
|---|---|---|---|---|---|---|
| 1 | 7:45–7:53 | KPI 总卡（黑底，3 列） | 左：`累计 ↓ 16.4%`<br>中：`2 轮迭代 · 18 份 artifacts`<br>右：`人工介入次数 = 0` | — | 三列等宽，数字 96pt | sustain |
| 2 | 7:53–7:58 | 架构图淡入 + 标语 | — | 标语：`LLM 驱动 · 闭环可审计 · 跨域自动协同` | — | swell |
| 3 | 7:58–8:00 | 平台 Logo + 内部项目地址 | — | — | — | BGM 渐出 |

---

## 附录 A — 字幕卡完整文案（共 11 张，依出现顺序）

| # | 出现 SCENE | 主标（英） | 副标（中） | 备注 |
|---|---|---|---|---|
| C00 | 1.1 | `HM-VERIF Kernel LLM 优化平台` | `闭环 Pipeline 自动迭代演示` | 主标改中文 |
| C01 | 2.1 | `Stage 1 — Intake & Routing` | `入站与路由` | `01 / 07` |
| C02 | 2.2 | `Stage 2 — Research` | `调研` | `02 / 07` |
| C03 | 2.3 | `Stage 3 — Plan Review (GATE)` | `计划评审 — 不通过即回退` | `03 / 07`，警示边 |
| C04 | 2.4 | `Stage 4 — Implementation` | `实现` | `04 / 07` |
| C05 | 2.5 | `Stage 5 — Code Review (GATE)` | `代码评审 — 不通过即回退` | `05 / 07`，警示边 |
| C06 | 2.6 | `Stage 6 — Tester A/B Validation` | `真机 A/B 验证` | `06 / 07` |
| C06b | 2.6 BEAT 5 | （仅中文，全屏一行） | `切换镜像中...` | 1.0s 中插卡 |
| C07 | 4.1 | `Stage 7 — Decision & Memory` | `决策与长期记忆` | `07 / 07` |
| C08 | 5.1 | `Iteration #2 — Closed-Loop Re-Entry` | `第 2 轮迭代自动启动 · 无人介入` | 节奏更急 |
| C09 | 5.4 | （静态卡） | `LLM 驱动 · 闭环可审计 · 跨域自动协同` | 收尾标语 |

---

## 附录 B — 屏幕注释短语库（按用途）

**Agent 身份（浅蓝 #7CD4FD）**：
- `agent: kernel-pipeline-starter v1`
- `agent: os-opt-manager v1`
- `agent: {specialist} v1`（如 `wq-threadpool-opt v1` / `memmgr-reclaim-research v1`）
- `agent: kernel-plan-reviewer v1`
- `agent: kernel-code-agent v1`
- `agent: kernel-code-reviewer v1`
- `agent: kernel-tester-agent v1`

**关键 artifact（黄 #FFC700）**：
- `优化候选 plan`
- `计划评审`
- `代码补丁`
- `代码评审`
- `A/B 验证报告`
- `目标长期记忆`
- `已采纳想法库`

**Verdict & 状态（绿 #2EC27E / 红 #E5484D）**：
- ✓ `通过，pipeline 自动进入实现阶段`
- ✓ `通过，进入真机 A/B 验证`
- ✓ `BUILD PASS`
- ✓ `IMPROVED — 指令数 ↓ N.N%`
- ✗ `驳回，回退到上一阶段`（仅备用 take）
- ✗ `BUILD FAIL`（仅备用 take）

**节奏/时间提示（屏角白字）**：
- `× 4 加速`
- `× 8 加速 · build 实际耗时 7m 12s`
- `× 16 加速 · 第 2 轮实际耗时 25m`

**结构性注释（屏底字幕条）**：
- `Hub-and-Spoke · 仅 manager 持 delegate 权 · 所有 sub-agent 完成后回 manager`
- `零人工介入 · 全部按约定路径落盘`
- `累计两轮闭环 · 零人工代码改动`

---

## 附录 C — 音效 cue 表

| 名称 | 用途 | 频次 | 取材建议 |
|---|---|---|---|
| `bass_drop_open` | 0:00 数字砸入 | 1 | freesound CC0 / 自制 sub |
| `bgm_main` | 全片 BGM（一首贯穿） | 1 | lofi ~85 BPM |
| `stinger` | 每个 stage 字幕卡前 | 7 | 短促电子 sweep |
| `gate_stinger` | 仅 GATE 字幕卡（C03 / C05） | 2 | 低频警示音 |
| `gate_pass_chime` | verdict APPROVED 瞬间 | 2 | 高频 ding |
| `gate_fail_thud` | 备用 take 用 | 0 | 低频钝音 |
| `tick` | 数字定格、字段脉冲、HUD 节点切换 | ~30 | 短促 click |
| `drop` | 指标 ↓ 大字弹出 | 2 | 重低音 hit |
| `whoosh` | 时间压缩切换、镜像切换 | ~6 | 风声 sweep |
| `success_chime` | BUILD PASS / artifact 落盘 | ~3 | 中频 bell |
| `data_stream` | MCP 响应高亮 | 1 | 数字流 |
| `swell` | macro view 拉近、outro 收尾 | 2 | 弦乐渐起 |
| `keyboard_asmr` | 用户敲命令、coder 编辑 diff | 2 | 真实机械键盘 |
| `flash_progress` | 手机刷机进度 | 2 | 低频 tone |
| `drum_fill` | Round 2 转场 | 1 | 鼓 fill |

---

## 附录 D — 终端 mock 文本对照（应贴在剪辑笔记里）

> **重要**：脚本中 SCENE 2.x 各 BEAT 给出的终端文本是**期望形态**，最终成片必须使用真实跑批的终端输出。如果真实输出与示意稿出入较大，按以下优先级处理：
>
> 1. **必须保留的行**：所有 `=== {agent} vN — ... ===` banner 行、所有 `verdict:` 行、所有 `tool_call:` 行、所有 `[manager] dispatching → ...` 行。
> 2. **可以删减/折叠的行**：研究阶段的 candidate 枚举、build 阶段的中间 `CC {X}.o` 行、刷机阶段的中间进度细节。
> 3. **绝对禁止改写的行**：任何指标数字、任何 verdict 字面值、任何 artifact 路径。

如果某行真实输出过长，**只截屏裁剪而不在剪辑里手敲替换**——保持"这是真跑出来的"的视觉信誉。

---

## 附录 E — 拍摄/剪辑 checklist（与 plan §6 互补，逐场用）

每个 SCENE 拍摄完，对照以下打勾：

- [ ] 该 SCENE 字幕卡素材已就位（附录 A 对照）
- [ ] 该 SCENE 全部屏幕注释短句已确认（附录 B 对照）
- [ ] HUD 当前激活节点正确
- [ ] 必须保留的终端行（附录 D）在原始素材中清晰可见
- [ ] 音效 cue 已在剪辑笔记标 t 值（附录 C 对照）
- [ ] PiP 子源（如有）已分别独立录到独立文件
- [ ] 时间码与 sync clap 对齐（误差 ≤ 1 帧）

---

*本剧本为拍摄/剪辑指导，不修改任何产品代码。所有终端文本示意稿仅用于规定镜头焦点，正式成片以真实跑批输出为准。*
