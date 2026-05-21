# HM-VERIF Kernel LLM 优化平台 — 闭环 Pipeline 汇报视频实施方案

> 目标观众：管理层 / 客户 / 内部技术评审
> 时长档位：**6–10 分钟（基准 8 分钟）**
> 表达形式：**无旁白 + 中文字幕卡 + 屏幕注释**
> 素材策略：**一次真实端到端运行 + 多源同步录制 + 事后裁剪**
> 交付分支：`claude/pipeline-demo-video-plan-SgbTF`

---

## 0. TL;DR

一句话计划：选一个能跑完全部 7 个 stage、整体耗时可控（30–90 min）的 `generic_full` 小目标，在 Windows 上用 OBS 同时抓 4 路画面（OpenCode 终端、MCP 服务栈、手机 scrcpy 镜像、HMOPT REST 指标看板），开跑前做 pre-flight 演练把所有变量锁死，跑完后用 DaVinci Resolve 按 7 个 stage 切片、对长任务做时间压缩与图示桥接，最终成片 8 分钟左右。

四条不可妥协的硬约束：

1. **一次"录全跑"必须可重放**：把"开跑前的环境"做成快照（git tag + 索引缓存 + 设备 stock 镜像），失败时可低成本重跑。
2. **无旁白意味着画面自己要会讲故事**：每个 stage 必须有一个"可视化变化点"（artifact 生成、agent 切换 banner、metrics 数字跳动）。
3. **MCP 协作必须看得见**：MCP 不是黑盒——给每个 MCP 一个固定的终端窗格 + 颜色 hue，让观众看到"哪个 MCP 在响应"。
4. **闭环要看得见**：至少录到第 2 轮迭代的指标对比，否则"自动闭环"无从证明。

---

## 1. 目标与受众

### 1.1 必须传达的信息（按优先级）

| # | 信息点 | 视觉证据 |
|---|---|---|
| P0 | **这是一个自动闭环**，不是脚本拼贴 | manager 自动 fan-out 到 7 个 agent，stage 间无人介入；第 2 轮迭代自动启动 |
| P0 | **跨 4 个域协同**（OpenCode/MCP/Windows/手机） | 同屏 4 分区始终可见 |
| P0 | **有 hard gate**（plan review / code review） | reviewer agent 给出 verdict，gate 不通过会回退 |
| P1 | **真机 A/B 验证**，指标真实下降 | 手机刷 stock → 跑 workload → 刷 feature → 跑 workload → metrics 数值对比 |
| P1 | **可解释**：每一步都有 artifact 落盘 | `.opencode/plans/`、`.opencode/reviews/`、`.opencode/patches/`、`.opencode/bench/` 文件树滚动 |
| P2 | **长期记忆**：bad-plan / memory accumulation 在生效 | `memory/targets/*.md` 文件增长 timelapse |

### 1.2 显式不展示

- 模型 API key / endpoint 细节（隐藏环境变量）
- 内部代码细节（patch 用 syntax highlight 看轮廓即可，不逐行讲）
- 失败案例的完整堆栈（保留"红色 verdict + 简短摘要"即可）

---

## 2. 核心叙事策略（无旁白怎么讲清楚）

无旁白条件下，节奏由三件事承载：

1. **章节字幕卡（chapter card）**：每个 stage 起始 1.5 秒全屏卡，黑底白字 + 一行副标题，例如：
   ```
       Stage 3 ──── Plan Review (GATE)
       计划评审 — 不通过即回退
   ```
2. **屏幕注释（call-out）**：用 DaVinci Fusion / 后期注释层做：
   - 黄色描边框圈出新生成的 artifact 文件
   - 红色↓箭头标注 metric 下降数字
   - 浅蓝色"agent: kernel-plan-reviewer v1"标签贴在终端 banner 行上
3. **音乐与音效**：
   - 底层 BGM：轻量电子 / lofi 鼓机，~85 BPM，stage 切换处做音乐 stinger
   - 关键音效：gate 通过/驳回（系统提示音）、metric 数字跳动（tick 音效）、build 成功（短促 chime）
   - 全程无人声

> 设计原则：观众即使**静音播放**也能看懂；BGM 只是节奏感的助推。

---

## 3. 目标选择与 demo-mode 准备

### 3.1 pipeline profile 选择

| profile | 走 tester | 是否适合本次 demo | 备注 |
|---|---|---|---|
| `sync_review` | ❌ | 不适合 | 不走真机 A/B，无法证明闭环 |
| `generic_full` | ✅ | **推荐** | 自动路由 + 全 7 stage + A/B，叙事自然 |
| `workqueue_full` | ✅ | 备选 | specialist 走 wq 路径，叙事更聚焦，但要选 workqueue 类 target |
| `hyperhold_full` / `memmgr_reclaim_full` | ✅ | 备用 | 体积偏大，build/flash 时间偏长 |

**首选：`generic_full` + 一个紧凑、构建快、有现成 hotspot 的 target。**

### 3.2 target 选择硬指标

需在 dry-run 中确认目标满足：

- 单次完整构建 ≤ 8 min（增量构建 ≤ 3 min）
- 单次刷机 ≤ 5 min（含 reboot 稳定等待）
- 单次 workload ≤ 3 min
- research/plan/review agent 端到端 ≤ 15 min
- 总和 1 轮迭代 ≤ 35 min；2 轮 ≤ 75 min

→ 留出≥30% buffer，**目标 1 轮 ≤ 25 min，2 轮 ≤ 55 min**。

### 3.3 demo-mode 配置清单（开跑前必须落实）

| 项 | 操作 | 验证方式 |
|---|---|---|
| 索引预热 | 提前跑 `hmopt index-kernel` + `hmopt index-runtime` | `outputs/index/` 已有 cache |
| MCP 服务全就位 | `bash scripts/run_all_mcp_servers.sh` 提前 10 min 启动 | `ss -ltn` 检查 7331/7333/7335/7336 |
| 真机 stock 镜像就绪 | 在 Windows 工作机准备 stock.img | `flash-device-operations.md` 流程演练过 |
| 真机 feature 镜像可即时构建 | build MCP 单跑 OK | `scripts/test_build_mcp.sh` 通过 |
| REST 看板 | `bash scripts/run_api.sh` 起在 :8000，浏览器开 `/runs/{id}/metrics` | 看板能看见数字刷新 |
| OpenCode 已 staging 好 prompt | `bash scripts/run_opencode_pipeline.sh --profile generic_full --target <X>` | `.opencode/state/current_prompt.md` 已生成 |
| git tag 锁定起点 | `git tag demo-take-1-start && git tag demo-baseline-clean` | 失败可回退 |
| `.opencode/state/` 清空 | 移到 `state.bak/` | 录到的是"全新一跑" |
| 显示器分辨率统一 | 4K 主屏 + 手机 scrcpy 1080p 镜像 | OBS 预览 16:9 |
| 屏蔽通知 | Windows 勿扰、IDE 通知关、IM 全静音 | 录前 10 min 检查 |
| 时间显示统一 | OBS 顶栏插一个 UTC 时钟源 | 后期对齐 4 路素材 |

---

## 4. 8 分钟分镜表（可伸缩到 6–10 分钟）

> 时间单位 = `mm:ss`。带 ★ 的段落是"必保镜头"；带 ◇ 的段落是弹性段，时长可压缩 / 拉长。
> 每段右侧"画面/字幕/音效"三栏说明无旁白下的视觉表达。

### 0. Cold Open / Title — `00:00–00:25` ★

| 时长 | 画面 | 字幕卡 / 屏幕注释 | 音效 |
|---|---|---|---|
| 6s | 黑屏 → 大字号数字快闪："7 stages · 4 domains · 2 gates · 1 closed loop" | 中文字幕卡："一次真实端到端运行" | 低频 bass drop |
| 8s | 4 路画面快速分屏闪现：OpenCode 终端 / MCP 终端 / 手机 scrcpy / 指标看板 | 每路画面右下小标签："OpenCode" / "MCP" / "Device" / "Metrics" | BGM 起 |
| 11s | 平台 Logo + 中文主标题："HM-VERIF Kernel LLM 优化平台 — 闭环 Pipeline 自动迭代演示" | — | sustain |

### 1. Pipeline Overview Animation — `00:25–00:55` ◇

| 时长 | 画面 | 字幕卡 / 屏幕注释 | 音效 |
|---|---|---|---|
| 30s | Motion Canvas / After Effects 制作的架构动画：从 `intake` 节点逐个亮起到 `decision`，hub `os-opt-manager` 在中心，7 个 agent 围绕；连线随话术依次点亮；末尾箭头从 `decision` 回到 `intake` 形成闭环 | 顶部字幕条："计划评审与代码评审是两道硬闸门，未通过即回滚" | tick 音 × 7（每个 stage 亮起时） |

> 这段是**唯一一段非真实录屏**，是用动画把架构讲清楚。后续 7 个 stage 的字幕卡可以呼应这张图。

### 2. Stage 1 — Intake & Routing — `00:55–01:30` ★

| 时长 | 画面 | 字幕卡 / 屏幕注释 | 音效 |
|---|---|---|---|
| 5s | 字幕卡："Stage 1 ─ Intake & Routing / 入站与路由" | — | stinger |
| 12s | 全屏录屏：用户敲下 `bash scripts/run_opencode_pipeline.sh --profile generic_full --target <X>`，回车 | 黄色框圈住 `--profile generic_full` 与 `--target` | 键盘 ASMR |
| 8s | `.opencode/state/current_prompt.md` 弹出，OpenCode 接管，`kernel-pipeline-starter` banner 出现 | 浅蓝标签贴 banner 行："agent: kernel-pipeline-starter v1" | — |
| 10s | manager `os-opt-manager` 接力 banner + 第一个 handoff packet 输出（目标、metric、hot path） | 右侧浮窗：handoff packet 5 个字段高亮 | — |

> 关键：让"无人介入即自动切换 agent"被看见。

### 3. Stage 2 — Research — `01:30–02:15` ★

| 时长 | 画面 | 字幕卡 / 屏幕注释 | 音效 |
|---|---|---|---|
| 5s | 字幕卡："Stage 2 ─ Research / 调研" | — | stinger |
| 15s | 主画面 OpenCode 显示 specialist（如 `wq-threadpool-opt`）调研，**子画面 PiP** 显示 MCP 终端中查询响应（main MCP 走 LlamaIndex + clangd）的命中流 | OpenCode 一侧框出"调用 mcp__hmopt__query"；MCP 一侧框出对应 RPC 行 | 数据流 whoosh |
| 15s | 文件管理器显示 `.opencode/docs/<target>_design.md` 和 `.opencode/plans/<target>_plan.md` 弹出 | 黄框圈两个新文件；中文注释："research 输出 → 喂给 plan-reviewer" | "ping" |
| 10s | 时间压缩段（4× 速度）：扫过调研日志中的关键命中（call graph、hotspot ranking） | 屏角字幕："× 4 加速" | — |

### 4. Stage 3 — Plan Review (GATE) — `02:15–02:55` ★

| 时长 | 画面 | 字幕卡 / 屏幕注释 | 音效 |
|---|---|---|---|
| 5s | 字幕卡："Stage 3 ─ Plan Review / 计划评审 ─ **GATE**" | — | stinger（gate 专属音色） |
| 15s | `kernel-plan-reviewer` banner → 评审过程录屏（带 instruction-count 评分） | 浅蓝标签："agent: kernel-plan-reviewer v1"；侧栏中文："硬闸门：未通过则回退到 Research" | — |
| 15s | `.opencode/reviews/<X>_plan_review.md` 文件展开，verdict 行被高亮（"APPROVED" 绿色） | 绿色对勾 ✓ 动画 | gate-pass chime |
| 5s | manager 接到 verdict 后自动 dispatch 到 coder（无人介入） | 字幕条："✓ 通过，pipeline 自动进入实现阶段" | — |

> 备选录制：开跑前在 **dry-run** 录一次"驳回"的 take 作为对照（章节 8 备用插入）。

### 5. Stage 4 — Implementation — `02:55–03:50` ★

| 时长 | 画面 | 字幕卡 / 屏幕注释 | 音效 |
|---|---|---|---|
| 5s | 字幕卡："Stage 4 ─ Implementation / 实现" | — | stinger |
| 15s | `kernel-code-agent` banner → 实时输出 diff（OpenCode 一侧），底部 PiP 显示 build MCP 终端正在 `make`（Windows 主机或容器） | OpenCode 一侧框出"修改文件 N 个"；build MCP 一侧框出"BUILD STARTED" | 键盘+编译 hum |
| 20s | **build 时间压缩（8× 速度）**：进度条快进，编译日志滚动 | 屏角字幕："× 8 加速 · build 实际耗时 7m 12s" | 加速音 |
| 15s | 编译成功 → `.opencode/patches/<X>.patch` 落盘 → `git diff --stat` 弹出 | 黄框圈 patch 文件；绿色 ✓ "BUILD PASS" | success chime |

### 6. Stage 5 — Code Review (GATE) — `03:50–04:30` ★

| 时长 | 画面 | 字幕卡 / 屏幕注释 | 音效 |
|---|---|---|---|
| 5s | 字幕卡："Stage 5 ─ Code Review / 代码评审 ─ **GATE**" | — | stinger |
| 20s | `kernel-code-reviewer` 输出评审，重点是 instruction-count tradeoff 段 | 高亮 "Instruction Count Tradeoff" 区块 | — |
| 10s | `.opencode/reviews/<X>_code_review.md` 展开 → APPROVED | 绿色对勾 | gate-pass chime |
| 5s | manager 自动进入 tester 阶段 | 字幕条："✓ 通过，进入真机 A/B 验证" | — |

### 7. Stage 6 — Tester A/B on Phone — `04:30–05:50` ★（视频最重头）

> 整段强制保留四分屏：左上 OpenCode、右上 build/flash MCP 终端、左下手机 scrcpy、右下 metrics 看板。

| 时长 | 画面 | 字幕卡 / 屏幕注释 | 音效 |
|---|---|---|---|
| 5s | 字幕卡："Stage 6 ─ Tester A/B Validation / 真机 A/B 验证" | — | stinger |
| 15s | **flash stock**：手机 scrcpy 显示 fastboot/recovery 界面，flash MCP 在 RPC 推 stock 镜像，进度条 | scrcpy 一侧字幕："Stock Image" | flash 进度音 |
| 15s | **workload on stock**：scrcpy 显示 workload 跑起来（界面动起来），右下指标看板第一列数字定格 | 红字"Baseline"贴在指标看板第一列 | tick 数字音 |
| 5s | 字幕卡（中插）："切换镜像中..."（时间压缩点） | — | swoosh |
| 15s | **flash feature** + workload：同样的画面节奏，指标看板第二列数字定格 | 绿字"Feature"贴在指标看板第二列 | — |
| 15s | **A/B diff** 动画：第三列出现，数字下降，红色↓箭头标注百分比下降 | 大字号 KPI："指令数 ↓ N.N%"；并行展示其它 metric | drop 音效 |
| 10s | `.opencode/bench/<X>_validation.md` 弹出 | 黄框 | — |

> 这一章节是最容易出问题的（设备掉线、刷机失败、workload 不稳定）。**必须在 dry-run 中跑通 ≥ 2 次。**

### 8. Stage 7 — Decision & Memory — `05:50–06:30` ◇

| 时长 | 画面 | 字幕卡 / 屏幕注释 | 音效 |
|---|---|---|---|
| 5s | 字幕卡："Stage 7 ─ Decision & Memory / 决策与记忆" | — | stinger |
| 20s | manager 接收 tester verdict，写决策 + 写 `.opencode/memory/targets/<X>.md` | 文件树侧栏滚动，新增/修改的 memory 文件被绿色↑标记 | — |
| 15s | 文件树 timelapse：本轮新增的所有 artifacts（plans / reviews / patches / bench / memory）依次亮起 | 注释："本轮共产出 N 份可审计 artifacts" | — |

### 9. Iteration Close-Up — `06:30–07:30` ★

| 时长 | 画面 | 字幕卡 / 屏幕注释 | 音效 |
|---|---|---|---|
| 5s | 字幕卡："闭环再启 ─ 第 2 轮迭代自动启动" | — | stinger（更高昂） |
| 15s | manager 再次 fan-out（前面节奏的快剪复用），新一轮 research banner 弹起 | 浅蓝标签 | — |
| 20s | **极速时间压缩（16×）**：第 2 轮 7 个 stage 的关键 banner / artifact 在 20 秒内闪过 | 屏角字幕："× 16 加速" | 加速音 |
| 20s | 指标看板出现**双柱对比图**：Baseline / Iter 1 / Iter 2 三列，曲线下降 | 大字号 KPI："累计指令数 ↓ M.M%" | drop 音效 |

> 这是"自动闭环"最有说服力的一段。如果第 2 轮 dry-run 风险大，可以**预生成 iter-2 数据**用同样的看板形态呈现，并在字幕卡注明"以下数据来自完整跑批，画面经时间压缩重现"。

### 10. Outro — `07:30–08:00` ★

| 时长 | 画面 | 字幕卡 / 屏幕注释 | 音效 |
|---|---|---|---|
| 10s | KPI 总卡：累计 instruction count ↓ / build 次数 / patch 数 / artifact 总数 / 人工介入次数 = 0 | 三行卡片样式 | sustain |
| 10s | 平台架构图淡入 → 标语 | 标语："**LLM 驱动、闭环可审计、跨域自动协同。**" | — |
| 10s | Logo + 项目地址（如内部 git） | — | BGM 渐出 |

---

## 5. 多源同步录制方案（OBS 场景设计）

### 5.1 物理布局

| 录制端 | 角色 | 硬件 |
|---|---|---|
| Windows 工作机 | 主录制端，跑 OpenCode + MCP + REST 看板，OBS 抓全屏 | 4K 主显（OBS 输出 1920×1080）、外接键盘麦不录人声 |
| 手机 | 真实被测设备 | USB 连 Windows，scrcpy 镜像入 OBS source |
| （可选）副屏 | 仅做"OBS 监视器"显示预览 | 不录 |

### 5.2 OBS 场景与源（scene = video chapter）

为每个 stage 准备一个 OBS scene，但**实际录制中不要用 OBS 实时切场景**——OBS 一直录单一"全画面"场景，所有切换都在后期 DaVinci 里做。这样素材是连续的，时间戳对齐零成本。

录制源（一律 1080p 30fps）：

1. **Source A — 全屏 Display Capture**：抓整个主显（开发者把 OpenCode、MCP 终端、文件管理器按象限固定布局）
2. **Source B — scrcpy Window Capture**：手机镜像（独立录一份，4 分屏时 PiP 使用）
3. **Source C — Browser Source（指标看板）**：`http://localhost:8000/runs/{id}/metrics` 的展示页（建议另起一个 demo 用 HTML 包装，把数字放大、加渐变色，刷新频率 1s）
4. **Source D — OBS 内嵌时钟**：屏幕角落显示 UTC + 录制相对时间（剪辑时用来对齐）

### 5.3 主显象限布局（开跑前用脚本一键定位窗口）

```
+--------------------------+--------------------------+
|                          |                          |
|  OpenCode 终端           |  MCP 服务多窗格           |
|  (1920x540, 上半区左)     |  (1920x540, 上半区右)     |
|                          |  · main MCP              |
|                          |  · build MCP             |
|                          |  · auto-test MCP         |
|                          |  · git MCP / seq MCP     |
+--------------------------+--------------------------+
|                          |                          |
|  文件管理器 + diff 视图   |  REST /metrics 看板      |
|  (1920x540, 下半区左)     |  (1920x540, 下半区右)     |
|                          |                          |
+--------------------------+--------------------------+
```

> 终端窗格用 Windows Terminal 多标签或 tmux + WSL。颜色 hue 区分：main=cyan、build=orange、auto-test=magenta、git=green、seq=violet。

### 5.4 录制规格

- 分辨率：1920×1080（最终成片同分辨率，避免缩放损失）
- 帧率：30 fps（30 帧足够终端 demo；高速度时间压缩仍清晰）
- 编码：x264，CRF 18，关键帧 2s
- 音频：32-bit float 单声道，48kHz；现场无环境音，**仅录 OBS 系统声**（便于后期做音效）；麦克风不录
- 文件命名：`take{N}_{YYYYMMDD-HHmm}_main.mkv`、`..._phone.mkv`、`..._dash.mkv`

### 5.5 多源时间对齐

每路素材录制时，OBS 同时启动；**开跑前在 4 路画面同步出现一个 sync clap**（按下 `Win+Shift+L` 触发屏幕闪烁 / 蜂鸣脚本）。后期以这个闪烁帧为 t=0。

---

## 6. 实拍 SOP（Take-1 执行流程）

### 6.1 录制前 60 分钟

- [ ] 关闭所有 IM、邮件客户端通知；勿扰模式 ON
- [ ] 显示器亮度固定，关闭夜间模式
- [ ] 跑 `scripts/run_all_mcp_servers.sh`，等 60s，`ss -ltn` 校验 4 个端口
- [ ] 跑 `bash scripts/run_api.sh &`，浏览器开 `/health`，看 OK
- [ ] 手机连 USB，`scrcpy --max-size=1080` 启动，确认输入响应正常
- [ ] 手机刷 stock 镜像至最近一次"已知干净"快照
- [ ] 创建/确认 git tag `demo-take-1-start`
- [ ] 清空/备份 `.opencode/state/` 与 `outputs/runs/*` 中本次目标的残留
- [ ] OBS 4 路源各跑一遍 10 秒测试，回放确认无掉帧

### 6.2 录制前 5 分钟

- [ ] OBS 开始录制（按 F9）
- [ ] 触发 sync clap 脚本
- [ ] 录"假启动" 30 秒空镜（让所有窗格静止）作为后期 cold-open 素材
- [ ] 屏息

### 6.3 录制中（即跑即录，不停录）

- [ ] 敲 `bash scripts/run_opencode_pipeline.sh --profile generic_full --target <X>` → 回车
- [ ] 全程不接触设备，**仅观察**
- [ ] 若 OpenCode 提示需要人工 approve 关键 gate（这是 harness 的设计点），按提示 approve；同时手不要遮挡屏幕
- [ ] 若任何 stage 在预设时长 1.5 倍仍未完成 → 在剪辑笔记里标记时间点，不中止录制
- [ ] 跑到第 2 轮 decision 完成后，再等 30 秒静止画面作为 outro 缓冲

### 6.4 录制后 30 分钟

- [ ] 停录，验证文件大小合理（≈ 1–2 GB/小时 × 4 路）
- [ ] 立刻把 4 路文件、`.opencode/` 全目录、`outputs/runs/<run_id>` 全部备份到独立磁盘 + 上传到内网对象存储
- [ ] 用 ffprobe 校验时长一致
- [ ] 把 sync clap 帧的时间戳记到剪辑笔记
- [ ] 创建 git tag `demo-take-1-end`

### 6.5 一次 take 失败的判定与重拍

**失败定义**（任意一条触发 → 重拍）：
- 任一 MCP 服务断连
- 手机刷机失败 / 设备掉线
- agent 路由错乱（manager 选错 specialist）
- 任一 stage 卡死 > 预设时长 × 2

**重拍前必做**：
- 回到 `git checkout demo-baseline-clean`
- 复原 stock 镜像
- 复盘失败点、列入 pickup 清单（章节 9.2）

允许的最大 take 数：**3 次**。3 次仍未一次性通过 → 启动"分段补拍"模式，重新规划。

---

## 7. 后期剪辑策略

### 7.1 时间压缩点位（Time Compression Map）

| 阶段 | 原始时长（估） | 压缩比 | 剪后时长 | 处理手法 |
|---|---|---|---|---|
| Research 调研日志 | ~10 min | × 4 | 2.5 min → 截 15s | speed-ramp + 关键命中减速回 1× |
| Implementation build | ~7 min | × 8 | ~52s → 截 20s | 进度条快进 + 编译尾段减速 |
| Flash stock / feature | 每次 ~4 min | × 6 | 每次 40s → 截 15s | 进度条全程加速，最后 reboot 减速 |
| Workload run | 每次 ~3 min | × 4 | 45s → 截 15s | 设备界面用恒速度，看板用恒速度同步 |
| Iteration 2（整轮） | ~25 min | × 16 | ~94s → 截 20s | 仅保留 7 个 agent banner 关键帧 |

> 速度变化处一定要有屏角字幕"× N 加速"提示——观众不能被瞒着加速。

### 7.2 剪辑工程组织（DaVinci Resolve）

- 一个 .drp 工程，时间线 1080p30
- 4 条 video track：V1 主画面、V2 PiP/scrcpy、V3 字幕卡 + 注释、V4 动画/motion graphic
- 3 条 audio track：A1 BGM、A2 音效、A3 系统声（极少使用）
- 用 markers 标出 7 个 stage 边界 + 时间压缩点 + sync clap

### 7.3 字幕卡 / 注释统一规范

- 字幕卡：1920×1080 黑底（#0B0B0F），主标题 96 pt，副标题 36 pt，字体 思源黑体 / 苹方
- 屏幕注释（call-out）：圆角矩形描边 4 px，颜色按用途：
  - 黄 #FFC700 = 关键 artifact 文件名
  - 绿 #2EC27E = 通过 / APPROVED / 指标改善
  - 红 #E5484D = 驳回 / FAIL / 指标退化
  - 浅蓝 #7CD4FD = agent 身份标签
- 字幕卡进出：opacity 0→100 over 8 frames，停留 1.0–1.4s，退场 6 frames
- 屏幕注释跟随画面元素出现/消失，**不允许飘字**

### 7.4 BGM / 音效

- BGM：licensed lofi 或 royalty-free 电子，约 85 BPM，整段一首；stage 切换不变曲，靠 stinger 制造节奏
- 关键音效（建议从 freesound.org 选 CC0）：
  - `stinger.wav` — 章节切换
  - `gate_pass.wav` — 闸门通过（中频 chime）
  - `gate_fail.wav` — 闸门驳回（低频钝音，本片仅 dry-run 备用素材用）
  - `tick.wav` — 数字跳动
  - `drop.wav` — 指标下降确认
  - `whoosh.wav` — 时间压缩切换

### 7.5 调色 / 终端可读性

- LUT：轻度 cinematic（保留终端高对比度），勿过度饱和
- 终端字号在原始录制时即放大到 14–16 pt，确保 1080p 下清晰
- 重要终端行做 **2.2× 局部放大 + 圆角裁切**（DaVinci Fusion 节点：Crop → Transform → Stroke）

---

## 8. 字幕卡 / 屏幕注释模板（供后期复用）

### 8.1 章节字幕卡（7 个 stage + cold open + outro）

每张卡的结构：

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│        Stage 3 ──── Plan Review (GATE)               │  ← 96pt 英文
│        计划评审 — 不通过即回退                          │  ← 36pt 中文
│                                                      │
│        03/07                                          │  ← 18pt 步骤计数
└──────────────────────────────────────────────────────┘
```

字幕卡文案稿（成片使用）：

| 序号 | 英文主标 | 中文副标 |
|---|---|---|
| 00 | HM-VERIF Kernel LLM Optimization | 一次真实端到端运行 |
| 01 | Stage 1 — Intake & Routing | 入站与路由 |
| 02 | Stage 2 — Research | 调研 |
| 03 | Stage 3 — Plan Review (GATE) | 计划评审 — 不通过即回退 |
| 04 | Stage 4 — Implementation | 实现 |
| 05 | Stage 5 — Code Review (GATE) | 代码评审 — 不通过即回退 |
| 06 | Stage 6 — Tester A/B Validation | 真机 A/B 验证 |
| 07 | Stage 7 — Decision & Memory | 决策与长期记忆 |
| 08 | Iteration #2 — Closed-Loop Re-Entry | 第 2 轮迭代自动启动 |
| 09 | Closed Loop, Audited, End-to-End | 闭环可审计、跨域自动协同 |

### 8.2 屏幕注释（call-out）短句库

中文为主，每条 ≤ 14 字，配合圆角描边框使用：

- 自动 fan-out，无人介入
- 硬闸门 — APPROVED 才能继续
- 新 artifact 落盘
- 真机已切至 stock 镜像
- 真机已切至 feature 镜像
- 指标对比 — Baseline vs Feature
- 时间压缩 × N
- 第 2 轮自动启动
- 长期记忆已更新

### 8.3 指标看板大字模板

`/runs/{id}/metrics` 页面在 demo 期间使用专用样式（建议加一个 `?demo=1` query 参数走简化布局）：

```
┌─────────────────────────────────────────────┐
│   PRIMARY METRIC — Instruction Count        │
│                                             │
│   Baseline       Feature        Δ           │
│   1,234,567   →  1,098,432   ↓ 11.0%        │
│                                             │
│   Secondary: cycles · cache-miss · wakeup   │
└─────────────────────────────────────────────┘
```

数字字体 96 pt，箭头与百分比绿色高亮。

---

## 9. 风险预案 & Pickup（补拍）清单

### 9.1 主要风险与对策

| 风险 | 触发概率 | 影响 | 预案 |
|---|---|---|---|
| 真机刷机失败 | 中 | 高（Stage 6 缺失） | 备一台同型号设备 + stock 镜像异机预热；最后一次失败接受 pickup |
| MCP 进程崩溃 | 低 | 中 | 录制前 10 min 启动并空跑一次；崩溃后立即重拍 |
| build 超长 | 中 | 中 | ccache + 索引预热；超 1.5 倍预设时长终止并重选 target |
| LLM 响应抖动（路由错） | 中 | 高 | 温度调低；prompt 中固定 specialist 提示；准备 fallback `workqueue_full` profile |
| 第 2 轮 metric 反而上升 | 低 | 高 | 接受真实结果，**用字幕诚实标注**"本轮未改善，闭环识别并写入 bad-plan memory"——这反而强化"可审计"卖点 |
| OBS 录制掉帧 | 低 | 中 | 录前 10 min 测试；备 NVENC GPU 编码 fallback |

### 9.2 Pickup 拍摄清单（如果 Take-1 局部失败可单段补拍）

按"对剪辑可替换性"从高到低排序——前面的最容易事后补：

1. **Pipeline overview 动画**（章节 1）：纯动画，与录屏脱钩，最后做
2. **章节字幕卡 / 注释 / outro KPI 卡**：纯后期，最后做
3. **指标看板大字段**（章节 7 末尾、章节 9）：可用 `/runs/{id}/metrics` 历史数据回放一次单独录
4. **gate 驳回备用 take**（章节 4 备用）：dry-run 时单独录一次
5. **Implementation diff 视图**：单独录一次 `git show` / IDE diff 即可
6. **手机 scrcpy 段**：可在事后单刷一遍重录，不会影响 OpenCode 终端时间线（用 PiP 嵌入）
7. **Build MCP 长跑加速段**：单独编一次重录最容易

不可单独补拍的部分（一旦失败必须重跑整次 take）：

- agent 间的真实 handoff banner 顺序
- 第 1 轮到第 2 轮的"自动连续性"
- manager 自动 dispatch 的决策瞬间

---

## 10. 工具栈

| 用途 | 工具 | 备注 |
|---|---|---|
| 屏幕录制 | OBS Studio 30+ | 多源、独立录文件 |
| 手机镜像 | scrcpy 2.x | USB 直连，低延迟 |
| 终端 | Windows Terminal + WSL2 / tmux | 多窗格固定布局 |
| 浏览器看板 | Chrome（kiosk 模式） | 看板专屏 |
| 视频剪辑 | DaVinci Resolve 18+ | 多机位 + Fusion 注释；免费版够用 |
| 动画 | Motion Canvas（推荐）/ After Effects | 架构图与 banner 动画 |
| 字幕卡静态 | Figma / Affinity Designer | 出 PNG 序列贴 V3 轨 |
| 音效 | freesound.org（CC0 优先） | 见 7.4 列表 |
| BGM | YouTube Audio Library / Epidemic Sound（如有授权） | 一首贯穿 |
| 终端 banner 字体 | JetBrains Mono / Source Code Pro | 14–16pt |
| 字幕卡字体 | 思源黑体 / 苹方 | 中文显示 |

---

## 11. 时间表（按周）

> 以**录制日 D 日**反推。建议总周期 2 周。

### Week -2（D-14 ~ D-8）— 预生产
- D-14：本计划评审 + 目标 target 候选 3 个
- D-13：3 个候选目标各做 1 次 dry-run，记录耗时与稳定性
- D-12：最终目标定档；编写 demo-mode 一键启动脚本
- D-11：制作 Pipeline overview 动画（章节 1）+ outro KPI 卡（章节 10）模板
- D-10：录制 sync clap 脚本、看板 `?demo=1` 样式、Windows 窗格布局脚本
- D-9：完整 dry-run（不录），按 take SOP 演练
- D-8：完整 dry-run（录但不剪），评估时长是否在 6–10 min 范围

### Week -1（D-7 ~ D-1）— 拍摄
- D-7：第 2 次完整 dry-run，标定时间压缩点位
- D-5：**Take 1 正式录**
- D-4：复盘 Take 1，决定是否 Take 2
- D-3：（如需）Take 2
- D-2：素材整理 + pickup 拍摄（章节 9.2 中所有可拆段）

### Week 0（D-Day ~ D+5）— 后期
- D-Day：粗剪（rough cut）— 把全片切到 7 个 stage 边界
- D+1：精剪 — 时间压缩、PiP、对齐 sync clap
- D+2：注释与字幕卡（V3 轨）
- D+3：调色 + 音效 + BGM
- D+4：内部 review，1 轮修改
- D+5：交付

### 关键里程碑
- M1（D-12）：目标定档
- M2（D-8）：dry-run 通过，时长达标
- M3（D-3）：素材采集完成
- M4（D+4）：内部 review 通过
- M5（D+5）：成片交付

---

## 12. 交付物清单

成片：
- `demo_pipeline_v1.mp4` — 1080p30，~8 min，H.264/AAC（适合内部分享）
- `demo_pipeline_v1_4k.mp4` — 2160p30 备用（如客户要求）
- `demo_pipeline_v1_silent.mp4` — 无 BGM 版本（汇报现场用现场配音 / 现场讲解时使用）

附属：
- `demo_pipeline_v1_subtitles_zh.srt` — 中文字幕（即使无旁白也提供，方便回看时跳读）
- `demo_pipeline_v1_shotsheet.pdf` — 镜头分场表（章节 4 的导出 PDF，含每 shot 起止时间码）
- `demo_pipeline_v1_assets.zip` — 字幕卡 PNG / Logo / motion graphic 工程文件
- `demo_pipeline_v1_takes/`（内部存档） — 原始 4 路素材 + DaVinci .drp + 剪辑笔记

工程：
- 本计划 markdown 文档（即本文件）
- demo-mode 启动脚本（建议放 `scripts/demo_pipeline_prep.sh`，本计划不做）
- OBS 场景 collection 导出（建议放 `tools/demo/obs_scene.json`，本计划不做）

---

## 附录 A — Shot List 速查表（开机即用）

| # | t 起 | t 止 | 时长 | 主画面 | PiP | 字幕卡 | 注释 |
|---|---|---|---|---|---|---|---|
| 1 | 0:00 | 0:06 | 6s | 数字快闪 | — | 7 stages · 4 domains · 2 gates · 1 closed loop | — |
| 2 | 0:06 | 0:14 | 8s | 4 路快剪 | — | 一次真实端到端运行 | 四角标签 |
| 3 | 0:14 | 0:25 | 11s | Logo + 主标 | — | HM-VERIF Kernel LLM Optimization | — |
| 4 | 0:25 | 0:55 | 30s | 架构动画 | — | （内嵌） | 7 个 tick |
| 5 | 0:55 | 1:00 | 5s | 字幕卡 | — | Stage 1 — Intake & Routing | — |
| 6 | 1:00 | 1:12 | 12s | OpenCode 终端 | — | — | 框出 --profile / --target |
| 7 | 1:12 | 1:20 | 8s | OpenCode 终端 | — | — | 浅蓝 agent 标签 |
| 8 | 1:20 | 1:30 | 10s | OpenCode 终端 | handoff packet 浮窗 | — | 5 字段高亮 |
| 9 | 1:30 | 1:35 | 5s | 字幕卡 | — | Stage 2 — Research | — |
|10 | 1:35 | 1:50 | 15s | OpenCode | main MCP 响应 | — | 黄框 |
|11 | 1:50 | 2:05 | 15s | 文件树 + plan.md | — | — | 双黄框 |
|12 | 2:05 | 2:15 | 10s | research log × 4 | — | × 4 加速 | — |
|13 | 2:15 | 2:20 | 5s | 字幕卡 | — | Stage 3 — Plan Review (GATE) | — |
|14 | 2:20 | 2:35 | 15s | OpenCode reviewer | — | — | 浅蓝 + 侧栏中文 |
|15 | 2:35 | 2:50 | 15s | review.md verdict | — | — | 绿对勾 ✓ |
|16 | 2:50 | 2:55 | 5s | OpenCode dispatch | — | ✓ 通过，pipeline 自动进入实现阶段 | — |
|17 | 2:55 | 3:00 | 5s | 字幕卡 | — | Stage 4 — Implementation | — |
|18 | 3:00 | 3:15 | 15s | OpenCode diff | build MCP | — | "BUILD STARTED" |
|19 | 3:15 | 3:35 | 20s | build MCP × 8 | — | × 8 加速 · build 实际耗时 ... | — |
|20 | 3:35 | 3:50 | 15s | patch + git diff --stat | — | — | 绿对勾 BUILD PASS |
|21 | 3:50 | 3:55 | 5s | 字幕卡 | — | Stage 5 — Code Review (GATE) | — |
|22 | 3:55 | 4:15 | 20s | code_review.md | — | — | tradeoff 段高亮 |
|23 | 4:15 | 4:25 | 10s | verdict APPROVED | — | — | 绿对勾 |
|24 | 4:25 | 4:30 | 5s | dispatch | — | ✓ 通过，进入真机 A/B 验证 | — |
|25 | 4:30 | 4:35 | 5s | 字幕卡 | — | Stage 6 — Tester A/B Validation | — |
|26 | 4:35 | 4:50 | 15s | 四分屏 / scrcpy 主 | flash MCP | — | 红字 "Stock Image" |
|27 | 4:50 | 5:05 | 15s | 四分屏 / scrcpy 主 | metrics 看板 | — | "Baseline" |
|28 | 5:05 | 5:10 | 5s | 字幕卡（中插） | — | 切换镜像中... | — |
|29 | 5:10 | 5:25 | 15s | 四分屏 / scrcpy 主 | flash MCP | — | 绿字 "Feature" |
|30 | 5:25 | 5:40 | 15s | 大字 KPI | — | 指令数 ↓ N.N% | drop |
|31 | 5:40 | 5:50 | 10s | validation.md | — | — | 黄框 |
|32 | 5:50 | 5:55 | 5s | 字幕卡 | — | Stage 7 — Decision & Memory | — |
|33 | 5:55 | 6:15 | 20s | manager + memory file | — | — | 绿色↑ |
|34 | 6:15 | 6:30 | 15s | 文件树 timelapse | — | 本轮共产出 N 份可审计 artifacts | — |
|35 | 6:30 | 6:35 | 5s | 字幕卡 | — | 闭环再启 — 第 2 轮迭代自动启动 | — |
|36 | 6:35 | 6:50 | 15s | manager fan-out | — | — | 浅蓝 |
|37 | 6:50 | 7:10 | 20s | 7 stage × 16 | — | × 16 加速 | — |
|38 | 7:10 | 7:30 | 20s | 双柱对比图 | — | 累计指令数 ↓ M.M% | drop |
|39 | 7:30 | 7:40 | 10s | KPI 总卡 | — | （内嵌 KPI） | — |
|40 | 7:40 | 7:50 | 10s | 架构图 + 标语 | — | LLM 驱动、闭环可审计、跨域自动协同 | — |
|41 | 7:50 | 8:00 | 10s | Logo | — | — | BGM 渐出 |

---

## 附录 B — Dry-run 检查表（每次完整演练对照）

- [ ] 4 个 MCP 端口监听（main 7331 / seq 7333 / build 7335 / auto-test 7336）
- [ ] REST API /health 返回 OK
- [ ] OpenCode 能 staging prompt 且 banner 正确
- [ ] 手机 scrcpy 镜像无掉帧，输入响应
- [ ] index-kernel / index-runtime cache 命中
- [ ] 1 轮迭代实测时长 ≤ 25 min
- [ ] 2 轮迭代实测时长 ≤ 55 min
- [ ] 指标看板数字与 `outputs/runs/<id>/metrics.json` 一致
- [ ] 所有 artifact 文件按约定路径生成
- [ ] OBS 4 路素材时间戳一致（误差 ≤ 1 帧）

---

## 附录 C — 一句话 fallback

若 Take-1 / Take-2 / Take-3 仍无法一次跑通真实端到端，立即切换为**"主线真实录屏 + 真机 A/B 段单独补拍"** 的混合模式（最初被否决的策略 C），并对应延后交付 2 个工作日。这是兜底，不是首选。

---

*本文档为视频实施方案，不修改任何产品代码。*
