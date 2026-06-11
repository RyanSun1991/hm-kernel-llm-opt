# Team Skill Hub — 全景机制图解 / Full Mechanism Atlas (中英双语 CN·EN)

> 一图看懂：角色 Roles · 动作 Actions · 目录 Directories · 代码 Code · 门禁 Gates · 闭环 Loop
> 配套交互版：`docs/skill_hub_flow_diagram.html`（浏览器直接打开 open in browser）

---

## 图 1 · 全景闭环 / The Big Picture Closed Loop

成员照常工作 → 一条命令蒸馏 → git PR 投稿 → 中央仓质检/策展/发版 → 下一个人自动受益。
Members work as usual → one distill command → contribute via git PR → hub gates/curation/release → the next member benefits automatically.

```mermaid
flowchart TB
    classDef dir fill:#fff7e6,stroke:#d48806,color:#613400
    classDef code fill:#e6f4ff,stroke:#1677ff,color:#002c8c
    classDef gate fill:#fff1f0,stroke:#cf1322,color:#5c0011
    classDef human fill:#f9f0ff,stroke:#722ed1,color:#22075e
    classDef out fill:#f6ffed,stroke:#389e0d,color:#135200

    subgraph M["① 成员侧 Member side · 每位工程师电脑 each engineer's machine"]
        direction TB
        M1["👤 成员 Member<br/>照常用 .opencode/ manager / research 入口干活<br/>work as usual via the .opencode multi-agent harness"]:::human
        M2[("📁 正源 Source of truth · .opencode/memory/<br/>idea_ledger/⟨target⟩.md — L00x 机制裁决台账 verdict ledger<br/>global_lessons.md — 通用教训 reusable lessons<br/>targets/ · subsystems/ — 目标/子系统记忆 notes")]:::dir
        M3["⚙️ 新增的桥 The new bridge<br/>hmopt sediment-opencode --bundle<br/>代码 code: src/hmopt/sediment/opencode_reader.py + pipeline.py<br/>markdown 解析 parse → JSON-Schema 校验 validate"]:::code
        M4[("📦 本地暂存 local staging (gitignored)<br/>.opencode/local/sediment_staging/_bundle.jsonl<br/>schema 合规候选 schema-valid candidates")]:::dir
        M1 -->|"沉淀 accumulate · 习惯零改动 zero habit change"| M2
        M2 --> M3
        M3 --> M4
    end

    M4 ==>|"② 投稿 Contribute<br/>cp → hm-skill-hub/staging/⟨member⟩/⟨date⟩.jsonl + git PR"| H0

    subgraph H["③ 中央仓 Central hub · hm-skill-hub/ · git PR 驱动 PR-driven"]
        direction TB
        H0[("📁 staging/⟨member⟩/*.jsonl<br/>候选收件箱 candidate inbox")]:::dir
        H1["🚦 CI 门禁 PR gates · .github/workflows/skill-hub-ci.yml<br/>tools/lint.py schema+路径一致性 path-scope check<br/>tools/redact.py 脱敏 secret scan<br/>tools/dedup.py --check 查重/冲突 dup-conflict (冲突即红 fail on conflict)<br/>tools/eval_gate.py 技能回归 skill regression"]:::gate
        H2["🧠 策展 Curation · 人 human + 引擎A engine A<br/>tools/central_curate.py --report report.md<br/>按报告人工定稿 manual finalize → 分配稳定 id assign stable ids"]:::human
        H3[("📁 knowledge/ 定稿知识 curated knowledge<br/>global/ heuristics · bad_plans · anti_patterns · validation_pitfalls<br/>targets/⟨slug⟩/ facts · idea_ledger")]:::dir
        H4["🌙 夜间闭环 Nightly loop · tools/nightly.py [--apply]<br/>收集 collect → 规范 normalize → 聚类 cluster(引擎A)<br/>→ 优化 optimize(引擎B skill_optimizer.py) → eval门 gate<br/>→ 发版 release.py semver → 广播 broadcast.py"]:::code
        H5["🎓 晋升检测 Promotion · tools/promotion_detector.py<br/>同机制 ≥2 独立实例 same mechanism, ≥2 distinct instances<br/>→ 只开建议 PR suggest-only · 永不自动合 never auto-merge"]:::gate
        H6[("📁 skills/ 技能库 · core · domain · technique<br/>⟨name⟩/SKILL.md + scorecards/ 评分卡 + eval/ 测试集 suites")]:::dir
        H0 --> H1
        H1 --> H2
        H2 --> H3
        H3 --> H4
        H3 --> H5
        H5 --> H6
        H4 --> H6
    end

    H4 ==>|"④ 发版广播 Release broadcast<br/>registry.yaml + .opencode/skill-memory.lock 钉版本 pin version"| C1

    subgraph C["⑤ 消费/回流 Consume and feed back · 任何成员 any member"]
        direction TB
        C1["🔎 读路径 Read path<br/>hmopt resolve ⟨path::symbol⟩ --stage research<br/>代码 code: src/hmopt/skillhub/resolver.py + retrieval.py<br/>selector匹配子系统 → 技能闭包 requires → 混合检索 hybrid retrieval"]:::code
        C2["🧩 挂载的上下文 Mounted context<br/>该函数的事实 facts + 坑 bad plans + 教训 lessons<br/>按阶段 token 预算裁剪 trimmed to stage budget<br/>审计 audit: retrieval.jsonl"]:::out
        C1 --> C2
    end

    C2 ==>|"喂给下一次优化 session feeds the next run · 经验越用越准"| M1
```

---

## 图 2 · 写路径映射 / Write-path Mapping（`.opencode/memory/` → hub 四大记录族 four record families）

`sediment-opencode` 的解析规则一览。模板里 HTML 注释中的示例行自动忽略 / template examples inside HTML comments are ignored.

```mermaid
flowchart LR
    classDef src fill:#fff7e6,stroke:#d48806,color:#613400
    classDef rec fill:#f6ffed,stroke:#389e0d,color:#135200

    S1["📄 idea_ledger/⟨target⟩.md<br/>每行 row: ### L001 ⟨机制一句话 mechanism⟩<br/>+ status / delta_pct / compare_level / rationale 字段 bullet"]:::src
    S2["📄 global_lessons.md<br/>每条 entry: ### ⟨教训标题 lesson title⟩<br/>+ kind / applies_when / do_or_dont / tags"]:::src
    S3["📄 targets/⟨slug⟩.md<br/>## Known Bad Plans 小节 section"]:::src
    S4["📄 targets/⟨slug⟩.md<br/>## Stable Structural Facts<br/>## Good Optimization Directions"]:::src
    S5["📄 subsystems/⟨name⟩.md<br/>同上小节 same sections"]:::src

    R1["🧾 idea · L9xx<br/>状态机裁决 verdict: approved/landed/<br/>reverted/rejected/deferred<br/>related_ids 保留原 L00x 溯源 provenance"]:::rec
    R2["🧾 global_lesson · H9xx/A9xx/V9xx<br/>id 前缀按 kind 匹配 schema 约束<br/>H=heuristic A=anti_pattern V=validation_pitfall"]:::rec
    R3["🧾 bad_plan · B9xx<br/>别人不再踩的坑 the trap others skip"]:::rec
    R4["🧾 memory_item fact · F9xx<br/>function 级 scope + target_slug"]:::rec
    R5["🧾 memory_item fact · F9xx<br/>subsystem 级 scope"]:::rec

    S1 --> R1
    S2 --> R2
    S3 --> R3
    S4 --> R4
    S5 --> R5
```

---

## 图 3 · 中央仓双引擎治理 / Two Governance Engines

### 引擎 A：知识合并（追加式 append-only）/ Engine A: knowledge merge

```mermaid
flowchart TB
    classDef gate fill:#fff1f0,stroke:#cf1322,color:#5c0011
    classDef ok fill:#f6ffed,stroke:#389e0d,color:#135200

    IN["📥 incoming 候选 candidate"] --> REL{"相关吗 related?<br/>同 scope + 词面重叠 lexical overlap<br/>或机制别名命中 or mechanism alias hit"}
    REL -->|"否 no"| NEW["✅ new 新增 add<br/>(新实例/新条件/新口径 additive)"]:::ok
    REL -->|"是 yes"| POL{"结论相反 opposite verdict?<br/>且同条件 same applies_when?"}
    POL -->|"是 yes"| CONF["⛔ conflict 冲突 → CI 红 fail<br/>tools/conflict_resolve.py 双时态消解<br/>旧记 superseded 墓碑 tombstone · 绝不物理删除 never delete"]:::gate
    POL -->|"否 no"| SIM{"相似度 similarity ≥ 0.82<br/>且同 scope same scope?"}
    SIM -->|"是 yes"| MER["🔗 merge 合并出处 provenance<br/>confirmations++ · 不新增 id no new id<br/>(喂 ≥2 确认晋升规则 feeds promotion rule)"]:::ok
    SIM -->|"否 no"| NEW
```

### 引擎 B：技能优化（eval 门 monotone gate）/ Engine B: skill optimization

```mermaid
flowchart LR
    classDef gate fill:#fff1f0,stroke:#cf1322,color:#5c0011
    classDef ok fill:#f6ffed,stroke:#389e0d,color:#135200

    E1["📜 技能文本 skill text<br/>= 冻结模型的外部参数<br/>external params of a frozen model"] --> E2["tools/skill_optimizer.py<br/>竞争式候选编辑 competing edits"]
    E2 --> E3{"留出 eval 集 holdout suite:<br/>严格变好 strictly better<br/>且零回归 zero regression?"}
    E3 -->|"是 yes"| E4["✅ 接受 accept<br/>scorecard 新版本 new scorecard<br/>dashboard 趋势 ▲"]:::ok
    E3 -->|"否 no"| E5["⛔ 拒绝 reject → bad_edits 缓冲<br/>下次跳过 skipped next time<br/>(反喂安全 anti-poisoning)"]:::gate
```

---

## 图 4 · 读路径内部 / Inside the Read Path（`hmopt resolve`）

```mermaid
flowchart TB
    classDef code fill:#e6f4ff,stroke:#1677ff,color:#002c8c
    classDef out fill:#f6ffed,stroke:#389e0d,color:#135200

    Q["🔎 hmopt resolve mm/vmscan.c::shrink_node --stage research"] --> P1["1️⃣ split_target<br/>path = mm/vmscan.c · symbol = shrink_node"]:::code
    P1 --> P2["2️⃣ selector 匹配 match<br/>_registry/subsystem_selectors.yaml<br/>path_globs + symbol_selectors → mm-reclaim<br/>(内核 rebase 只改这张表 only this table changes on rebase)"]:::code
    P2 --> P3["3️⃣ 技能闭包 skill closure<br/>domain/mm-reclaim → requires →<br/>technique/hoist-loop-invariant + core/*"]:::code
    P3 --> P4["4️⃣ 混合检索 hybrid retrieval<br/>BM25(符号名强) + 哈希向量(模糊词面) · RRF 融合<br/>scope 预过滤 prefilter + hub 知识 maturity ≥ L2 门"]:::code
    P4 --> P5["5️⃣ 阶段预算裁剪 trim to stage budget<br/>research 8条/3000tok · plan 5/2000 · implement 3/1500<br/>(防上下文污染 anti context-pollution)"]:::code
    P5 --> OUT["🧩 输出 output: skills + knowledge<br/>+ retrieval.jsonl 逐次审计 per-call audit"]:::out
```

---

## 图 5 · 一条经验的生命周期 / Lifecycle of One Piece of Knowledge

```mermaid
stateDiagram-v2
    direction LR
    s0: L0 本地草稿 local draft
    s1: L1 候选 candidate
    s2: L2 稳定 stable
    s3: L3 确认 confirmed
    sup: superseded 被取代·墓碑保留 tombstone kept
    sk: 🎓 technique 技能 skill

    [*] --> s0: .opencode/memory/ 沉淀 accumulate
    s0 --> s1: hmopt sediment-opencode 蒸馏 distill
    s1 --> s2: PR 门禁 gates + 策展定稿 curate → knowledge/
    s2 --> s3: 跨成员确认 cross-member confirmations
    s2 --> sup: 冲突且新证据更强 stronger contradiction (不删 never deleted)
    s2 --> sk: 同机制 ≥2 实例 promotion (建议制 suggest-only)
    s3 --> sk
```

---

## 表 1 · 角色 × 动作 × 入口 / Role × Action × Entry Point

| 角色 Role | 时机 When | 动作/入口 Action / Entry | 看什么结果 What to observe |
|---|---|---|---|
| 👤 成员 Member | 研究新函数前 before researching a target | `hmopt resolve "⟨path::symbol⟩" --stage research --run-dir .opencode/state` | 挂载的 skills+knowledge 列表；`retrieval.jsonl` 审计 |
| 👤 成员 Member | 任务/session 收口 at close-out | `hmopt sediment-opencode --opencode-dir .opencode --hub hm-skill-hub --contributor ⟨你⟩ --bundle` | CLI 打印 `N valid / M invalid / parse notes`；`_bundle.jsonl` 内容 |
| 👤 成员 Member | 投稿 contribute | `cp _bundle.jsonl hm-skill-hub/staging/⟨你⟩/⟨日期⟩.jsonl` + git PR | CI 四道门 绿/红 |
| 👤 成员 Member | 共享流程技能 share a skill | `hmopt promote-skill .opencode/skills/⟨name⟩ --kind core\|domain\|technique` | 生成的 L0 脚手架 + 待补步骤清单 next steps |
| 🧠 维护者 Maintainer | 收到投稿 PR on contribution PR | `python tools/central_curate.py staging/⟨m⟩/⟨d⟩.jsonl --report report.md` | 逐条 add/merge/conflict 决策报告 |
| 🧠 维护者 Maintainer | 定稿 finalize | 手工把候选写成 `knowledge/...` 的 md（分配稳定 id）→ `python tools/lint.py` | `OK — N record file(s) validated` |
| 🧠 维护者 Maintainer | 每晚/每周 nightly | `python tools/nightly.py` → 复核后 `--apply` | 7 步报告；`registry.yaml` 版本；`skill-memory.lock` |
| 🧠 维护者 Maintainer | 找晋升机会 promotions | `python tools/promotion_detector.py --pr-body` | promote-candidate PR 文案 |
| 🤖 CI（自动 auto） | 任何 PR any PR | `skill-hub-ci.yml`（hub 门禁+消费测试）· `core-smoke.yml`（import/CLI/全量测试） | GitHub Checks 绿/红 |
| 📊 所有人 Everyone | 看健康度 health | `python tools/dashboard.py` → `eval/scorecards/_dashboard.md` | 每技能 pass_rate 版本趋势（▲=改进） |

## 表 2 · 目录与代码地图 / Directory & Code Map

| 路径 Path | 是什么 What it is | 谁写 Written by | 谁读 Read by |
|---|---|---|---|
| `.opencode/memory/idea_ledger/` | 机制裁决台账（正源）verdict ledgers (source of truth) | 成员/harness | `sediment-opencode` |
| `.opencode/memory/global_lessons.md` | 通用教训 lessons | 成员/harness | `sediment-opencode` |
| `.opencode/memory/targets\|subsystems/` | 目标/子系统记忆 notes | 成员/harness | `sediment-opencode` |
| `.opencode/local/sediment_staging/` | 本地暂存（gitignored）local staging | `sediment-opencode` / 收口钩子 hook | 成员（投稿时 cp） |
| `.opencode/skill-memory.lock` | hub 版本钉 pin | `tools/broadcast.py` | 消费侧 consumers |
| `src/hmopt/sediment/opencode_reader.py` | **桥**：memory markdown → 候选 the bridge | — | `sediment_opencode()` |
| `src/hmopt/sediment/pipeline.py` | 蒸馏编排 + bundle 重编 id | — | CLI / 钩子 |
| `src/hmopt/sediment/skill_promote.py` | 成员技能 → hub L0 脚手架 | — | `hmopt promote-skill` |
| `src/hmopt/skillhub/resolver.py` | 读路径编排 read-path orchestration | — | `hmopt resolve` |
| `src/hmopt/skillhub/retrieval.py` | BM25+向量混合检索 hybrid retrieval | — | resolver / eval |
| `hm-skill-hub/staging/` | 候选收件箱 inbox | 成员 PR | CI 门禁 / curate / nightly |
| `hm-skill-hub/knowledge/` | 定稿知识（唯一真相源 markdown）curated knowledge | 维护者 | resolver / dedup / promotion |
| `hm-skill-hub/skills/` | 技能库 core·domain·technique | 维护者 / promote-skill | resolver / optimizer / eval_gate |
| `hm-skill-hub/schemas/` | 四大记录族 + 技能 frontmatter JSON-Schema | 维护者 | lint / sediment validate |
| `hm-skill-hub/_registry/subsystem_selectors.yaml` | target→子系统映射（rebase 只改这） | 维护者 | resolver |
| `hm-skill-hub/tools/` | 中央工具链 lint/redact/dedup/curate/nightly/... | — | CI / 维护者 |
| `hm-skill-hub/eval/` | 留出测试集 + scorecards + 看板 | run_evals / dashboard | eval_gate / 所有人 |

---

## 设计要点速记 / Design Tenets at a Glance

1. **两类资产两台引擎** Two assets, two engines — 知识"追加+查重+冲突消解"（绝不删历史）；技能"就地编辑+eval 严格变好"（反喂安全）。
2. **markdown 是唯一真相源** Markdown is the source of truth — 索引只是派生缓存，可随时重建；一切可 git 评审。
3. **成员习惯零改动** Zero habit change — 正源就是大家已经在写的 `.opencode/memory/`；新增的只是收口一条蒸馏命令。
4. **一切门禁皆 fail-loud** Gates fail loud — 冲突即 CI 红；晋升永远只建议、人来合。

---

## 图 6 · 知识落位路由 / Knowledge Placement Routing（候选 → `knowledge/` 哪个目录）

**规则（设计 §6.1 "路径即 scope"）**：每条记录一个文件；**文件路径编码 scope**，与 frontmatter 的 `scope` 字段冗余且必须一致——`tools/path_scope.py` 推导期望值，`tools/lint.py` 在 CI 强校验，不一致即拒。
Rule (design §6.1 "path encodes scope"): one record per file; the file path **encodes** the scope and must agree with the frontmatter — derived by `tools/path_scope.py`, CI-enforced by `tools/lint.py`.

```mermaid
flowchart LR
    classDef dir fill:#fff7e6,stroke:#d48806,color:#613400
    classDef hub fill:#e6f4ff,stroke:#1677ff,color:#002c8c

    C["📦 staging/⟨member⟩/⟨date⟩.jsonl 候选<br/>candidate = schema + frontmatter scope<br/>(策展人按此定稿落位 curator places by this)"]:::hub --> D{"按 schema + scope 路由<br/>route by schema + scope<br/>tools/path_scope.py 推导 · lint.py CI 强校验"}

    D -->|"global_lesson · kind=heuristic"| K1[("knowledge/global/heuristics/H###.md<br/>例 H001-hoist-before-inline")]:::dir
    D -->|"global_lesson · kind=anti_pattern"| K2[("knowledge/global/anti_patterns/A###.md<br/>例 A001-over-optimizing-cold-paths")]:::dir
    D -->|"global_lesson · kind=validation_pitfall"| K3[("knowledge/global/validation_pitfalls/V###.md<br/>例 V001-single-image-test")]:::dir
    D -->|"bad_plan · applies_to.subsystems=星号全局"| K4[("knowledge/global/bad_plans/B###.md<br/>例 B001-blanket-inline-kworker")]:::dir
    D -->|"bad_plan · applies_to=[⟨subsystem⟩]"| K5[("knowledge/subsystems/⟨sub⟩/bad_plans/B###.md")]:::dir
    D -->|"memory_item · scope.level=subsystem"| K6[("knowledge/subsystems/⟨sub⟩/⟨id⟩.md")]:::dir
    D -->|"memory_item · level=function/call-site/data-flow<br/>+ scope.target_slug"| K7[("knowledge/targets/⟨slug⟩/facts/F###.md<br/>例 F001 → targets/mm-vmscan-c-shrink-node/facts/")]:::dir
    D -->|"idea · target_slug"| K8[("knowledge/targets/⟨slug⟩/idea_ledger/L###.md<br/>例 L001 → targets/mm-vmscan-c-shrink-node/idea_ledger/")]:::dir
```

### 表 3 · 落位速查 / Placement Cheat-sheet（与 `path_scope.py` 逐行对应 line-for-line with code）

| 候选 Candidate（schema + 判别字段 discriminator） | 落位目录 Destination | id 前缀 | 仓内真实例子 Real example |
|---|---|---|---|
| `global_lesson` · `kind: heuristic` | `knowledge/global/heuristics/` | `H###` | `H001-hoist-before-inline.md` |
| `global_lesson` · `kind: anti_pattern` | `knowledge/global/anti_patterns/` | `A###` | `A001-over-optimizing-cold-paths.md` |
| `global_lesson` · `kind: validation_pitfall` | `knowledge/global/validation_pitfalls/` | `V###` | `V001-single-image-test.md` |
| `bad_plan` · `applies_to.subsystems: ["*"]` | `knowledge/global/bad_plans/` | `B###` | `B001-blanket-inline-kworker.md` |
| `bad_plan` · `applies_to.subsystems: [⟨sub⟩]` | `knowledge/subsystems/⟨sub⟩/bad_plans/` | `B###` | （布局已支持，暂无实例） |
| `memory_item` · `scope.level: subsystem` | `knowledge/subsystems/⟨sub⟩/` | `F/G/R###` | （布局已支持，暂无实例） |
| `memory_item` · `scope.level: function`/`call-site`/`data-flow` + `target_slug` | `knowledge/targets/⟨slug⟩/facts/`（或 `decisions/`） | `F###` | `targets/mm-vmscan-c-shrink-node/facts/F001-hoist-sc-priority.md` |
| `idea` · `target_slug` | `knowledge/targets/⟨slug⟩/idea_ledger/` | `L###` | `targets/mm-vmscan-c-shrink-node/idea_ledger/L001-hoist-sc-priority.md` |

**谁执行落位 Who places**：策展人 curator——dedup 判 `new` 时建**新文件**（分配下一个稳定 id）；判 `merge` 时**不建新文件**，把出处并进已有文件（`confirmations++`）；判 `conflict` 时先消解（旧文件标 `superseded`，新文件落位）。落错目录或 scope 写错 → `lint.py` CI 直接红。
The curator places files: `new` → new file with the next stable id; `merge` → no new file, provenance merged into the existing one; `conflict` → resolve first. A wrong directory or scope fails CI.

---

## 图 7 · 技能落位 / Skills Placement（§6.2 三层 + "比子系统细的都不是技能"）

**唯一原则 The one rule**：skill 树只按「种类/稳定性」分层；**dir/file/function 维度不是 skill，是 knowledge**，运行时由 selector 挂载——从根上消灭拓扑组合爆炸，rebase 只改一张 selector 表。
Skills are layered by kind/stability only; anything finer than a subsystem is **knowledge**, mounted at runtime by selectors — topology explosion eliminated, a kernel rebase touches one selector table.

```mermaid
flowchart LR
    classDef sk fill:#fff7e6,stroke:#d48806,color:#613400
    classDef kn fill:#f6ffed,stroke:#389e0d,color:#135200
    classDef sel fill:#e6f4ff,stroke:#1677ff,color:#002c8c

    D1["维度①流程/跨切面 process<br/>例: 先研究后实现、指令数优先"]
    D2["维度②优化招式 mechanism<br/>例: hoist-loop-invariant<br/>(按招式命名 · 不按 target)"]
    D3["维度③子系统 subsystem<br/>例: mm-reclaim"]
    D4["维度④目录 dir glob<br/>例: mm/star-reclaim-star"]
    D5["维度⑤文件 file<br/>例: mm/vmscan.c"]
    D6["维度⑥函数 function/symbol<br/>例: shrink_node"]

    D1 --> S1[("skills/core/⟨name⟩/SKILL.md<br/>例 core/instruction-count-first/")]:::sk
    D2 --> S2[("skills/technique/⟨mechanism⟩/SKILL.md<br/>例 technique/hoist-loop-invariant/")]:::sk
    D3 --> S3[("skills/domain/⟨subsystem⟩/SKILL.md<br/>例 domain/mm-reclaim/<br/>(skills 里唯一触及拓扑的层)")]:::sk
    D4 --> S4["不建目录 no folder!<br/>= domain skill 的 applies_to.path_globs<br/>+ _registry/subsystem_selectors.yaml 单点绑定"]:::sel
    D5 --> S5[("不是技能 not a skill!<br/>→ knowledge/targets/⟨slug⟩/facts/")]:::kn
    D6 --> S6[("不是技能 not a skill!<br/>→ knowledge/targets/⟨slug⟩/idea_ledger/<br/>+ selector 表的 symbol_selectors")]:::kn

    S2 -.->|"由晋升产生 born from promotion:<br/>同机制 ≥2 实例 landed → promotion_detector 建议<br/>知识实例留作证据 instances stay as evidence"| S5
```

**每个技能文件夹的解剖 Anatomy of a skill folder**（Anthropic SKILL.md 标准）：
`SKILL.md`（何时用+怎么用，选中即加载）· `best_skill.md`（引擎B SkillOpt 制品）· `evals/`（留出测试集）· `candidates/`（Pareto 前沿候选）· `scorecards/`（版本化评分卡）· `references/`（重材料按需加载）。

**两条进入 skills/ 的路 Two ways in**：
1. **知识毕业 Knowledge graduates**：`promotion_detector` 发现同 mechanism ≥2 独立实例 → 建议开 `skills/technique/⟨mechanism⟩/` PR（实例 fact 留在 `knowledge/` 作证据，不搬家）；
2. **成员技能提升 Member skill promoted**：`hmopt promote-skill .opencode/skills/⟨name⟩ --kind core|domain|technique` → 生成 L0 脚手架，补 eval+scorecard 后过门毕业。

### 一条记录的完整旅程（拿本仓真实 id 串起来）/ One record's full journey with real ids

```text
① .opencode/memory/idea_ledger/mm-vmscan-c-shrink-node.md   ← 成员正源（### L001 landed -0.8%）
② hmopt sediment-opencode --bundle
   → .opencode/local/sediment_staging/_bundle.jsonl          ← 候选 {"schema":"idea","id":"L901",related_ids:["L001"]}
③ cp → hm-skill-hub/staging/alice/2026-06-11.jsonl + PR      ← Tier-1 收件箱（还没进 knowledge/!）
④ CI: lint+redact+dedup --check                              ← dedup 判 new
⑤ 策展定稿（人）按 schema+scope 落位:
   idea     → knowledge/targets/mm-vmscan-c-shrink-node/idea_ledger/L001-hoist-sc-priority.md
   fact     → knowledge/targets/mm-vmscan-c-shrink-node/facts/F001-hoist-sc-priority.md
   教训     → knowledge/global/heuristics/H001-hoist-before-inline.md
   坑       → knowledge/global/bad_plans/B001-blanket-inline-kworker.md
   （路径↔frontmatter scope 一致性由 lint CI 把关）
⑥ 同机制第 2 个实例 landed → promotion_detector 建议
   → skills/technique/hoist-loop-invariant/SKILL.md          ← 知识毕业成技能（F001 留作证据 subsumed_by H001）
⑦ hmopt resolve "mm/vmscan.c::shrink_node"                   ← 下一个人同时拿到 F001+H001+B001 和 technique 技能
```
