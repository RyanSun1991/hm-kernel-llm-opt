# Team Skill Hub 操作手册（Runbook · 中文版）

> 每个环节：**谁做 · 何时做 · 敲什么命令 · 输入是什么 · 产出什么文件 · 产物流向哪里**。
> 所有命令均已实测。约定两个环境变量（按你的实际路径设置）：
>
> ```bash
> export HUB=~/work/hm-kernel-llm-opt-main/hm-skill-hub      # 中央仓
> export KOPEN=~/.../hm-verif-kernel/.opencode               # 你内核仓的 .opencode
> ```

## 链路总览（各环节编号，下文逐节对应）

```
┌─ 知识链路（引擎A·追加式）────────────────────────────────────────────┐
│ ①日常工作         ②蒸馏              ③投稿            ④CI门禁(自动)  │
│ .opencode/memory → _bundle.jsonl  →  hub staging/  →  lint·redact·   │
│ (+reviews/bench)   (本地暂存)         <你>/<日期>      dedup·eval_gate│
│                                                          ↓           │
│ ⑥晋升检测 ←─────  hub knowledge/  ←─────  ⑤策展定稿(人)              │
│ (同招式≥2实例)     (团队知识库)            curate报告→落位→分配id      │
└──────────┬───────────────────────────────────────────────────────────┘
           ↓ 建议毕业成 technique 技能
┌─ 技能链路（引擎B·eval门）──────────────────────────────────────────┐
│ ⑦晋升脚手架        ⑧出考卷+登记         ⑨考试+体检+PR               │
│ promote-skill  →  suite/cases+eval_id → run_evals→scorecard→lint    │
│ (L0实习生)         +best_skill.md        ≥及格线→合入                │
└──────────┬─────────────────────────────────────────────────────────┘
           ↓
┌─ 运营与消费 ───────────────────────────────────────────────────────┐
│ ⑩夜间闭环(维护者)              ⑪消费(任何成员·研究新函数前)          │
│ nightly dry-run→--apply        hmopt resolve "<路径::函数>"          │
│ 发版+更新lock+看板              → 自动挂载全队 事实+坑+招式 → 回到①    │
└────────────────────────────────────────────────────────────────────┘
```

## 环节 0 · 一次性部署

| | |
|---|---|
| 谁/何时 | 每位成员，装一次 |
| 操作 | `cd ~/work/hm-kernel-llm-opt-main && pip install -e ".[dev]"` |
| 验收 | `hmopt --help` 能列出 `sediment-opencode / resolve / retrieval-eval / promote-skill` |
| 说明 | 全离线可用，无需任何环境变量；`--llm-extract` 才需要 `HMOPT_LLM_API_KEY`/`HMOPT_LLM_BASE_URL` |

---

# 知识链路（你的经验 → 团队知识库）

## 环节 ① · 日常工作沉淀（不需要新命令）

| | |
|---|---|
| 谁/何时 | 成员 + harness agent，照常干活 |
| 操作 | 用 `.opencode/` 的 manager/research 入口照常跑优化；按 memory-accumulation 约定让结论落盘 |
| 产物 | `$KOPEN/memory/idea_ledger/<目标>.md`（`### L00x` 裁决行）· `memory/global_lessons.md`（`### 标题`+bullet）· `memory/targets|subsystems/*.md`（`## Known Bad Plans` 等小节）；兜底源：`reviews/*_review.md`、`bench/*_validation.md`、`state/*bad_plans*.md` |
| 流向 | 全部喂给环节 ② |
| ⚠️ 格式要点 | 台账行必须是 `### L001 一句话` + `- **status**: landed` 字段 bullet；HTML 注释里的模板示例不会被采集 |

## 环节 ② · 蒸馏（收口一条命令）

| | |
|---|---|
| 谁/何时 | 成员，任务/session 收口时 |
| 操作 | `hmopt sediment-opencode --opencode-dir "$KOPEN" --hub "$HUB" --contributor <你> --bundle`（可选 `--llm-extract --config <平台仓>/configs/app.yaml` 让 LLM 从 docs/plans 自由文本提炼） |
| 输入 | 环节 ① 的全部文件 |
| 产物 | `<opencode-dir>/local/sediment_staging/opencode-<仓名>.jsonl`（本批）+ **`_bundle.jsonl`（投稿物，每行一条 schema 合规候选）** |
| 验收 | 终端打印 `N valid candidate(s)`；0 时会列出扫描统计和期望格式 |
| 流向 | `_bundle.jsonl` → 环节 ③ |
| ⚠️ 常见坑 | 必须指向 `.opencode` 整个目录（不是 memory 子目录）；路径不存在会直接报错并给提示 |

## 环节 ③ · 投稿（git PR）

| | |
|---|---|
| 谁/何时 | 成员，蒸馏后自查 bundle 内容（vim 过一眼 LLM 提炼条目）确认想共享 |
| 操作 | `mkdir -p "$HUB/staging/<你>" && cp <bundle路径> "$HUB/staging/<你>/$(date +%F).jsonl"` → git 提交、推送、开 PR |
| 产物 | `hm-skill-hub/staging/<你>/<日期>.jsonl`（Tier-1 收件箱，**尚未进知识库**） |
| 流向 | PR 触发环节 ④ |

## 环节 ④ · CI 门禁（自动，无需操作）

| | |
|---|---|
| 谁/何时 | CI，PR 上自动跑（也可本地预跑） |
| 操作（本地预跑） | `cd "$HUB" && python tools/lint.py && python tools/redact.py --check && python tools/dedup.py staging/<你>/<日期>.jsonl --check` |
| 产物 | dedup 逐条三态判定（终端）：`merge`=并出处 / `conflict`=同条件相反结论·**CI 红必须先消解** / `new`=新增 |
| 验收 | GitHub Checks 全绿 |
| 流向 | 绿 → 环节 ⑤ |

## 环节 ⑤ · 策展定稿（维护者·人工）

| | |
|---|---|
| 谁/何时 | 维护者，收到投稿 PR 后 |
| 操作 | `python tools/central_curate.py staging/<你>/<日期>.jsonl --report report.md` → 按报告逐条把候选**手写成 md 文件**放进 knowledge/ 正确目录、分配正式 id → `python tools/lint.py` 确认 |
| 产物 | `report.md`（逐条 add/merge/conflict 建议）+ **knowledge/ 里的正式记录文件** |
| 落位规则（路径=scope，CI 强校验） | idea→`knowledge/targets/<slug>/idea_ledger/L###.md`；函数级 fact→`targets/<slug>/facts/F###.md`；通用教训→`global/heuristics|anti_patterns|validation_pitfalls/`（H/A/V）；全局坏招→`global/bad_plans/B###.md`；子系统级→`subsystems/<sub>/` |
| 流向 | 知识库更新 → 环节 ⑥（自动检测）和 ⑪（被检索到） |

## 环节 ⑥ · 晋升检测（知识→技能的自动桥）

| | |
|---|---|
| 谁/何时 | 维护者，定期（或 nightly 顺带） |
| 操作 | `python tools/promotion_detector.py --pr-body` |
| 产物 | promote-candidate PR 文案：同一招式在 ≥2 个独立目标实测有效 → 建议开 `skills/technique/<招式>/`（**只建议、人合并**；原知识记录留作证据不搬家） |
| 流向 | 人采纳 → 进入技能链路 ⑧⑨ 补考卷毕业 |

---

# 技能链路（你的 .opencode 流程技能 → 团队技能库）

## 环节 ⑦ · 晋升脚手架

| | |
|---|---|
| 谁/何时 | 成员，想共享某个流程技能时 |
| 操作 | `hmopt promote-skill .opencode/skills/<名字> --kind core --hub "$HUB"`（流程类一律 core；domain 需额外填 applies_to+selector 表，初期不碰；technique 一般由环节 ⑥ 产生） |
| 产物 | `$HUB/skills/core/<名字>/SKILL.md`（L0/experimental 占位）+ 终端打印毕业清单（即 ⑧⑨） |
| 验收 | `python tools/lint.py` 技能数 +1 |

## 环节 ⑧ · 出考卷 + 登记 + 写工作底稿

| | |
|---|---|
| 谁/何时 | 成员/维护者，技能想升 L1、想受门保护时（**L0 入库不强制**） |
| 操作 | ① 建 `eval/task_suites/<套件>/suite.yaml`（name/description/pass_threshold）+ `cases/*.yaml`（每题：`expected_terms` 好指导必提的要点 + `avoid_terms` 雷词 + weight）；② 在 SKILL.md frontmatter 加 `eval_id: eval/task_suites/<套件>`；③ 写 `best_skill.md`（要点操作清单，引擎B的优化对象） |
| 出题素材 | **从 knowledge/ 已验证记录出题**（最佳）或 LLM 起草+人审；**不要从技能文本自身生成**（循环论证） |
| 共卷规则 | 考点相同的技能可共用一套（`eval_id` 指同处）；考点不同必须分卷 |
| 产物 | 考卷目录 + 更新的 SKILL.md + best_skill.md |

## 环节 ⑨ · 考试 + 体检 + PR

| | |
|---|---|
| 谁/何时 | 成员，⑧ 完成后 |
| 操作 | `python tools/run_evals.py skills/core/<名字> --suite=eval/task_suites/<套件>` → `python tools/lint.py` → 开 PR |
| 产物 | `skills/core/<名字>/scorecards/<名字>__<版本>.json`（成绩单，重跑同版本会覆盖） |
| 验收 | `pass_rate ≥ pass_threshold` 且 lint 全绿 |
| 流向 | 合入后由环节 ⑩ 的引擎B接管持续优化 |
| ⚠️ 常见坑 | **手动跑必须带 `--suite=`**（不带会回落到默认考卷考出无意义的 0 分）；frontmatter 的 `eval_id` 是给自动门禁（eval_gate/nightly）用的，两处都要有 |

---

# 运营与消费

## 环节 ⑩ · 夜间闭环（维护者）

| | |
|---|---|
| 谁/何时 | 维护者，每晚/每周 |
| 操作 | `python tools/nightly.py`（dry-run 看 7 步报告）→ 人复核无误且**确有内容变更**时 `--apply` |
| 产物（--apply 时） | 引擎B接受的技能编辑 + 新 scorecard；`registry.yaml` 版本升级；`releases/<版本>.md`；**`.opencode/skill-memory.lock` 更新（消费端钉版本）**；`eval/scorecards/_dashboard.md` 看板（▲=改进趋势） |
| 验收 | 报告 7 行全 ok；`normalize`/`validate` 任一失败会自动 abort 不落盘 |

## 环节 ⑪ · 消费（闭环回到 ①）

| | |
|---|---|
| 谁/何时 | 任何成员，**研究新函数之前** |
| 操作 | `hmopt resolve "<路径::函数>" --stage research --run-dir .opencode/state`（在平台仓跑可自动发现 hub；在别处跑加 `--hub`） |
| 产物 | 终端：挂载的 skills + knowledge 列表（带分数/成熟度）；`--run-dir` 下追加 `retrieval.jsonl` 逐次审计（查"AI 当时看到了什么"） |
| 效果 | 你在环节 ⑤ 入库的经验，出现在**别人**的这份清单里 = 闭环完成 |

## 附 · 观测与门禁速查

| 看什么 | 命令/位置 |
|---|---|
| 技能健康度趋势 | `python tools/dashboard.py` → `eval/scorecards/_dashboard.md` |
| 检索引擎自检（改了检索代码才需要；成员日常不用跑） | `hmopt retrieval-eval --eval-dir "$HUB/eval/retrieval"`，低于基线 exit 1 |
| 某次检索审计 | `<run-dir>/retrieval.jsonl` |
| 知识/技能格式体检 | `python tools/lint.py`（只查格式，不评质量；质量归 run_evals） |

## 附 · 产物流向总表

| 产物 | 产生于 | 被谁消费 | 最终归宿 |
|---|---|---|---|
| `memory/idea_ledger` 等 md | ① harness | ② sediment-opencode | 留在成员本地（正源） |
| `_bundle.jsonl` | ② | ③ 你自己 cp | 用后即弃（gitignored） |
| `staging/<你>/<日期>.jsonl` | ③ | ④ CI · ⑤ 策展 | 定稿后可清理 |
| `knowledge/**/*.md` | ⑤ | ⑥ 晋升检测 · ⑪ resolve · ④ dedup 比对 | **永久（追加式，只打墓碑不删）** |
| `skills/<kind>/<名>/SKILL.md` | ⑦/⑥ | ⑪ resolve（经 requires/selector 挂载）· ⑩ 引擎B | 永久（就地编辑，受 eval 门保护） |
| `scorecards/*.json` | ⑨/⑩ | eval_gate 防回归 · dashboard | 按版本累积 |
| `skill-memory.lock` | ⑩ broadcast | 消费端钉 hub 版本 | 每次发版更新 |
| `retrieval.jsonl` | ⑪ | 人工审计 | 本地日志 |
