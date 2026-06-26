# Team Skill Hub — 设计与操作手册（中文）

> 团队级、版本化的"经验中枢"：让 AI 内核优化中积累的经验，能**跨成员积累、自动验证、安全复用**，形成 **消费 → 蒸馏 → 策展 → 门控 → 发布 → 再消费** 的自进化闭环。
>
> 本文覆盖：核心设计思路、目录 layout、交互流程、MCP 接入、以及每一步的**具体命令与执行方法**。配套英文版见 `Skill_Hub_Design_And_Operations_EN.md`。

---

## 1. 一句话与痛点

**一句话**：把"AI 优化内核"过程中积累的经验（哪些招式有效、哪些是坑、某函数该怎么改、什么验证不可信）外置到一个**带质量门、带版本号、防退化**的团队中央仓 `hm-skill-hub`，像用"私有 npm 包"一样消费，且越用越准。

**痛点**：经验只留在各人本地（`.opencode/memory`），跨人/跨项目/跨时间无法复用；重复探索、反复踩坑；而"简单共享"又会让知识自相矛盾、或让 AI 自我优化陷入"反馈自强化"（平均变好、个别做砸）。

---

## 2. 核心设计思路（关键决定）

### 决定 1：两类资产，两台引擎（最重要）
| | **知识 Knowledge（事实/教训）** | **技能 Skills（做法/流程）** |
|---|---|---|
| 例 | "shrink_node 里 sc->priority 重复读，hoist 省 0.8%" | "出方案前先查 bad_plans 去重"这套流程 |
| 怎么演化 | **只追加 + 七路关系分类合并**，永不物删，矛盾双时态并存 | **就地竞争式编辑**，受 eval 门保护 |
| 引擎 | **引擎 A**（集合并 / dedup / conflict / subsumption） | **引擎 B**（SkillOpt：有界编辑 + Pareto + eval 门） |

> 分开是为了：用同一套 git 行级合并治理两者 → 知识会自相矛盾、技能会被某人一周的坏经验覆盖。**两套引擎、两道门，分开走。**

### 决定 2：知识合并是"七路关系分类"，不是重复/矛盾二分
incoming 与已有记录分 7 类，**铁律：除"明确矛盾且新证据更强"外任何分支都不物删**：
`duplicate`(合出处) · `contradiction`(旧记 superseded 不删) · `temporal`(过时保留可审计) · `conditional`(条件不同共存) · `subsumption`(泛化包含，具体留作证据) · `selector`(路径变重解析) · `evidence`(同 delta 不同口径合并)。

### 决定 3：markdown 是唯一真相源
每条知识 = 一个 `.md`（YAML frontmatter + markdown 正文）。可 git 评审、可手改、索引后端可换；**jsonl 只是 staging 投稿的中转格式，落库即转 `.md`**。

### 决定 4：技能 = 可训练的外部参数（反喂安全）
技能文本当成冻结模型的"外部参数"。任何改动必须在留出 eval 集上**严格变好且零回归**才接受，否则进 `bad_edits` 缓冲。这是杜绝"反馈自强化"的根本机制。

### 决定 5：门控 ≠ 策展
- **门控（Gating）= 守门员**：自动、客观、二值（过/不过）。只问"合不合法/安不安全/会不会退步"，不懂内容好坏、不决定放哪。
- **策展（Curation）= 编辑/馆员**：人主导（工具辅助）。问"对不对/值不值得共享/跟旧的什么关系/放哪、给什么正式 ID"。

### 决定 6：通过 MCP 接入，不依赖本地 CLI
Agent 在内核仓里运行、`hmopt` 不在其 PATH 上，故经 **MCP 工具**触达 hub；任何 hub 调用失败**静默降级、绝不阻塞主流程**。

---

## 3. 目录 Layout

### 3.1 中央仓 `hm-skill-hub/`
```
hm-skill-hub/
├── registry.yaml              ★总目录/索引（版本、schema 指纹、技能清单），release.py 维护
├── knowledge/                 【知识】只追加，按 scope 落位（正式内容 = .md）
│   ├── global/{heuristics,anti_patterns,validation_pitfalls,bad_plans}/   H/A/V/B
│   ├── subsystems/<子系统>/    子系统级
│   └── targets/<函数slug>/{facts,decisions,idea_ledger}/   F### / L###
├── skills/{core,domain,technique}/<名>/   SKILL.md + best_skill.md + scorecards/*.json
├── staging/<成员>/<日期>.jsonl  ★投稿收件箱（Tier-1 候选，jsonl 中转，未入库）
├── schemas/*.schema.json      记录的"固定格子"（Phase 0.5 校验）
├── _registry/{subsystem_selectors,mechanisms}.yaml   路径/符号→子系统、受控招式词表
├── eval/{task_suites,retrieval,scorecards}/   技能考卷 / 检索题库 / 看板
├── policies/*.md              谁决定/依据什么（merge/promotion/auto_merge）
├── releases/<版本>.md         发版记录
└── tools/*.py + ci_local.sh   全套工具 + 本地 CI
```
**关键**：`knowledge/` 的目录结构本身编码"经验管多大范围"（global / subsystems / targets），CI 强校验路径↔frontmatter scope 一致。

### 3.2 消费端（内核仓）`.opencode/`
```
.opencode/
├── skill-memory.lock          钉住消费的 hub 版本（path/pin/hub_version）
├── memory/{idea_ledger,targets,subsystems,global_lessons.md}   本地经验（resolve 的 --local-memory 叠加源）
├── local/sediment_staging/    蒸馏临时落点（gitignored，<run>.jsonl + _bundle.jsonl）
├── hub/                        ★把 hub 挂这里（submodule），供 MCP 服务端发现
└── ...（agents / commands / skills / state）
```

### 3.3 记录格式（一文件一记录）
- ID 前缀 = 类型：**F**=fact、**H**=heuristic、**A**=anti_pattern、**V**=validation_pitfall、**B**=bad_plan、**L**=idea。
- **9xx 段是 sediment 临时号**；正式号（F001…）策展时分配、全仓唯一。
- frontmatter = 结构化字段（机器检索/校验）；markdown 正文 = 人话解释。

---

## 4. 交互流程（自进化闭环）

```
        ┌────────────  hm-skill-hub（团队中央仓·semver）────────────┐
        │   knowledge/（引擎A）          skills/（引擎B）            │
        └──▲──────────────────────────────────────────┬───────────┘
 ④发布/广播 │ release + broadcast → 更新 .opencode/hub + lock        │ ①消费 resolve
           │                                            ▼
 ③策展+门控 │  staging/ ◄── ②蒸馏 sediment ◄── 一次优化 run ◄── ①把知识挂成上下文
   (引擎A/B + CI门 + 双人复核)        (.opencode/memory+bench → 候选)
```

**6 个 hub 接入点（在 `.opencode` 流水线中）**：
- ★1 intake：读 `skill-memory.lock` 取版本
- ★2 research 前：`skillhub_resolve(stage=research)` 注入 `## Hub context`
- ★3 research 内：hub 作为 5-idea 漏斗的去重源 #8
- ★4 plan-review 前：`skillhub_resolve(stage=plan-review)` 给去重门
- ★5 decision：`skillhub_sediment` 蒸馏回流
- ★6 发版回流：broadcast 更新 `.opencode/hub` + lock → 回到 ★1

---

## 5. MCP 接入

平台起一个 **skill-hub MCP server**（Docker，端口 7338，compose 服务 `hmopt-skillhub-mcp`），暴露 3 个工具：

| MCP 工具 | 作用 |
|---|---|
| `skillhub_resolve(target, stage, [opencode_dir], [mechanism])` | **读**：返回 `## Hub context` 块（技能 + 知识 + 坏招） |
| `skillhub_sediment([opencode_dir], [contributor], [bundle])` | **写**：把 `.opencode/memory` 蒸馏成候选 + `_bundle.jsonl` |
| `skillhub_status([opencode_dir])` | 钉的 hub 版本 + 可达性 |

- **服务端**（非 agent）做 `.opencode/` 文件 I/O（挂载内核仓），故 sub-agent 文件沙箱对 hub 不构成限制——**任何 mcp-enabled agent 都能调**。
- `opencode_dir` 缺省取服务端 `HMOPT_SKILLHUB_OPENCODE_DIR`（compose 默认 `/workspace/kernel/.opencode`）。
- 同仓本地（装了 `hmopt`）可用 CLI 等价：`hmopt resolve … --local-memory .opencode/memory --run-dir .opencode/state` / `hmopt sediment-opencode --opencode-dir .opencode --bundle`。

---

## 6. 操作方式与具体执行

### 6.1 消费（读）
研究/计划前由 agent 调 `skillhub_resolve(target, stage)`，把返回的 `## Hub context` 块作为上下文；审计落 `.opencode/state/retrieval.jsonl`。

### 6.2 蒸馏（写）
优化 run 收口、本地 memory 写完后调 `skillhub_sediment(contributor=<你>, bundle=true)` → 产 `.opencode/local/sediment_staging/_bundle.jsonl`。
> 非空前提：本地 memory 用 sediment 认得的格式（`### L00x`+`status: landed`+`delta_pct`；`## Known Bad Plans`；review `## Decision reject`；bench `verdict: pass`+`delta_pct`）。

### 6.3 投稿到 hub（第一次 commit）
```bash
# ① 看 + triage（丢测试/LLM 元知识垃圾；坏 mechanism/截断标题先改）
cat .opencode/local/sediment_staging/_bundle.jsonl
# ② 放进 hub 收件箱
cd <hub>; mkdir -p staging/<成员>
cp <内核仓>/.opencode/local/sediment_staging/_bundle.jsonl staging/<成员>/$(date +%F).jsonl
# ③ 门控预检（绿了再提交）
bash tools/ci_local.sh
# ④ commit 到 hub
git add staging/<成员>/$(date +%F).jsonl && git commit -m "sediment: <成员> <日期>" && git push   # 或开 PR
```
此时是**收件箱待审件，尚未成为共享知识**。

### 6.4 策展定稿（curation，第二次 commit）
```bash
cd <hub>
python tools/central_curate.py staging/<成员>/<日期>.jsonl --plan     # 预览：分配正式 ID + 路径
python tools/central_curate.py staging/<成员>/<日期>.jsonl --apply    # 写入 knowledge/**/*.md（机械活自动）
#   作用：七路分类(add/merge/conflict/subsumption)、分配正式 ID、按 scope 落位、写 .md
#   conflict/merge 留人处理；内容质量（招式名/标题）若未修在此修
bash tools/ci_local.sh                                                # 门控复校（lint 新写的 md）
# + 双人复核（1 域 + 1 流程；bootstrap 可自审先合）
git add knowledge/ && git commit -m "curate: land <ids>" && git push
git rm staging/<成员>/<日期>.jsonl                                     # 收件箱可清（或留作溯源）
```
**到此**：进 `knowledge/` 并 push → 任何人对该 hub 跑 `resolve` 即可检索到 =（基础）共享。

### 6.5 发布/广播（带版本治理的正式共享）
```bash
cd <hub>
python tools/nightly.py     # 7 步闭环报告（dry-run）
python tools/release.py     # 升 semver（schema 变→major）
python tools/broadcast.py --hub-version=<v> --sha=<sha>   # 更新各消费仓 .opencode/hub + skill-memory.lock
```
下次别人的 research/plan `resolve` 自动挂上你这条 = **完整闭环共享**。

### 6.6 本地 CI（门控，无需 GitHub）
`tools/ci_local.sh` 精确镜像 hub CI 的 5 道门，本机或 Docker 一条命令跑全套：
```bash
bash tools/ci_local.sh
# 等价 5 步：pytest tools/tests/ · lint.py · redact.py --check · 遍历 staging dedup.py --check · eval_gate.py
docker compose run --rm hmopt bash -lc "bash hm-skill-hub/tools/ci_local.sh"   # Docker 内
```

---

## 7. 门控 vs 策展（分工速查）
| | 门控 Gating | 策展 Curation |
|---|---|---|
| 角色 | 守门员（机器/客观/二值） | 编辑馆员（人/判断/归档） |
| 处理 | schema·路径scope·id唯一·脱敏·**未消解冲突**·技能**eval退步** | **七路分类**·内容修正·分配正式ID·按scope落位·**双人复核** |
| 工具 | `ci_local.sh`（lint/redact/dedup/eval_gate） | `central_curate.py`（--report/--plan/--apply）+ 人 |
| 能否全自动 | 是 | 机械部分能（ID/路径/写文件），判断归人 |

> dedup 双重身份：`dedup --check` 是门控（冲突→exit 1）；其分类又是策展输入（central_curate 据此建议 add/merge/conflict）。

---

## 8. 成熟度阶梯与晋升（谁决定/依据）
| 级别 | 依据 | 落到哪 | 谁决定 |
|---|---|---|---|
| L0 草稿 | 本地未结构化 | `.opencode/local/` | 自己 |
| L1 候选 | schema 齐全 + 初步证据 | `staging/<成员>/<日期>.jsonl` | 自己 |
| L2 稳定 | 过三门 + 双人复核 | `knowledge/` 或 `skills/domain|technique/` | 1 域 + 1 流程 |
| L3 核心 | ≥2 子团队复用 + owner 签字 | `skills/core/` | 2 owners + CODEOWNERS |

- **知识→技能毕业**：同一招式在 ≥2 个独立目标实测有效 → `promotion_detector` 发信号 → 人采纳升为 technique 技能（原记录留作证据）。
- **版本**：`release.py` 确定性推断（schema 变→major）。

---

## 9. 安全与降级不变量
- **反喂安全**：技能改动严格变好且零回归才接受。
- **永不物删**：淘汰记录打 `superseded` 墓碑，审计链完整。
- **静默降级**：MCP/hub 不可达 → 返回 `hub: unavailable`，主流程不阻塞。
- **人把关**：入库需双人复核；自动合并须连续 ≥3 次改进且零回滚才解锁。

---

## 10. 诚实边界（当前状态）
- 头条数字（curator 1.0 / retrieval 1.0 / optimizer 0.67→1.00）**可复现，但是 fixture/proxy 构造结果**——证明"控制流/流水线打通"，非"真机能力"。真机 A/B 指令数 delta 是后续长杆。
- 投稿 PR 的人工评审、`--apply` 真改 hub、broadcast 真发版均为受控/半自动；sediment 写路径已接入但端到端需在装齐环境验证。
- 详细逐项真实度见 `docs/Session_Implementation_Review_CN.md`、`docs/Verify_And_Walkthrough_CN.md`。
