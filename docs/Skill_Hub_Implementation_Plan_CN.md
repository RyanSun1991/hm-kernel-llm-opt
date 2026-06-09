# Team Skill Hub 实现计划与细节设计

| 项 | 值 |
|---|---|
| 文档状态 | Draft v1.2（同步设计 v2.3 评审修订）|
| 日期 | 2026-06-09 |
| 关联设计 | `docs/Team_Skill_Hub_Design_CN.md`（v2.3）+ `docs/Team_Skill_Hub_Design_Diagrams_CN.md`（含读路径图）|
| 范围 | Phase 0–4 任务级分解；v1.2 把评审反馈（schema 阻塞 / retrieval 硬门 / 冲突分类 / subsumption）落到任务卡 |
| 语言策略 | 散文 zh-CN；schema/代码/CLI/commit 英文 |
| 修订 | v1.2（评审反馈）：① P0.5-2 升为 **Phase 1 前置阻塞 gate**（+ 路径编码 scope + CI 一致性 + schema 补 subsumes/applies_when）；② P1-8 PoC benchmark 扩为**七路冲突分类** + false-delete≈0 硬指标；③ P1-10 retrieval 升**硬门**（3 类 query + must/optional-hit + CI + 符号名 ablation）；④ Phase 2 加 P2-9 subsumption 检测器、喂 P2-8。v1.1：Phase 1 扩为 3–5w（读路径 + 本地两级），Phase 2 加 P2-8 晋升检测器 |

---

## 0. 总体节奏

```
Phase 0 抽取    1-2w   ┃ 本会话 ★
Phase 1 蒸馏    2-3w   ┃ 下会话
Phase 2 策展合并 3-6w   ┃
Phase 3 eval门  6-10w  ┃ 长杆
Phase 4 自动优化 10w+   ┃
```

**核心约束**：
- 全程在分支 `claude/tender-cray-ABIsw`；hub 初期作为本仓子目录 `hm-skill-hub/`，未来 `git subtree split --prefix=hm-skill-hub` 拆出独立仓。
- 每个 Phase 都有「Definition of Done」（DoD），不达成不进入下一阶段。
- 现有 `.opencode/{skills,agents,...}` **Phase 0 不动内容**，只建结构；内容迁移留到 Phase 0.5 专门会话做（带回归验证）。

---

## 1. Phase 0 — 抽取（本会话执行）

**目标**：零行为变更地把双仓骨架立起来；hub 能 lint 通过、能跑 CI 占位。

| ID | 任务 | 交付物（路径）| AC（验收）| 依赖 |
|---|---|---|---|---|
| P0-1 | 仓骨架 | `hm-skill-hub/{README,CONTRIBUTING,GOVERNANCE,CHANGELOG}.md`、`registry.yaml`、`.gitignore` | 6 文件存在、内容自洽 | — |
| P0-2 | 空目录占位 | `skills/{core,technique,domain}/`、`knowledge/{global/{lessons,anti_patterns},subsystems,targets,index}/`、`evidence/{benchmarks,regressions}/`、`eval/{task_suites,scorecards}/`、`staging/`、`releases/`（每个含 `.gitkeep`）| 目录树存在 | — |
| P0-3 | 7 份 JSON-Schema | `schemas/{bad_plan,global_lesson,memory_item,idea,skill_frontmatter,skill_patch,scorecard}.schema.json` | 每份是 valid JSON-Schema draft-07 | — |
| P0-4 | 控制词表 | `_registry/{mechanisms,subsystem_selectors}.yaml` | mechanisms 初始 ≥ 12 条（hoist/inline/batch 等）| — |
| P0-5 | 三份策略文档 | `policies/{promotion,merge,deprecation}_policy.md` | 固化设计 §8/§9/§10/§13 规则；可被人直接执行 | P0-3 |
| P0-6 | 解析器 + lint CLI | `tools/{parse_memory,lint,redact}.py`、`tools/requirements.txt` | `python tools/lint.py` 在空 hub 上 exit 0；带示例时验证 schema | P0-3 |
| P0-7 | CI 占位 | `.github/workflows/ci.yml`（hub 内）| 拆仓后即可激活 | P0-6 |
| P0-8 | 1 份示例 | `knowledge/global/anti_patterns/A001-*.md`、`knowledge/global/lessons/H001-*.md`、`skills/core/example/SKILL.md` | lint 通过；可作模板复用 | P0-3, P0-6 |
| P0-9 | 消费端占位 | `.opencode/skill-memory.lock`（pinning placeholder） | 格式自描述；hub 未拆前用 in-repo 模式 | — |

**DoD**：
- `python hm-skill-hub/tools/lint.py` 在含示例的 hub 上 exit 0。
- 目录树与设计 §6.1 一致。
- 任何团队成员能照 `CONTRIBUTING.md` + 示例文件 起手写一条 `bad_plan` 并通过 lint。

**Phase 0.5（独立小会话）—— 含 Phase 1 前置阻塞 gate**：

- **P0.5-1 内容迁移**——把现有 `.opencode/skills`、`agents`、`commands`、`pipelines`、`docs`（harness 规范部分）切入 `hm-skill-hub/`，在 `.opencode/` 下用 symlink 保持旧路径可用。带回归：跑一次现有 `/optimize_generic` 验证 pipeline 行为不变。
- **P0.5-2 schema / markdown 落盘格式收敛 ★前置阻塞**（v1.2 升级，评审反馈 ①，对应设计 §6.1 / §7 / §17 议题 7）：lint / dedup / retrieval scalar filter / Curator 七路分类全部依赖 schema 字段稳定，故**必须先于 Phase 1 完成**。交付：① **一记录一文件**（拆掉 `### A001` 多记录堆叠），frontmatter 用标准 schema 全字段，**禁止每类自扩字段**；② **文件路径编码 scope**（`knowledge/targets/<slug>/facts/<ID>.md` 等，§6.1）；③ `parse_memory.py` 升级为「frontmatter → 标准 schema object」转换器；④ `lint.py` 改 schema-driven + **新增路径 scope 与 frontmatter scope 一致性校验**；⑤ schema 补 `subsumes[]/subsumed_by[]/superseded_by[]/applies_when`；⑥ 现有内容一次性回填。**AC（gate）**：`python tools/lint.py` 对全部示例 + 任意新条目 exit 0；路径/frontmatter scope 不一致能精准报错；schema 含新字段；**此 gate 不过，Phase 1 不开工**。

---

## 2. Phase 1 — 蒸馏 + 读路径 + 本地在线消解（3-5w，v1.1 扩展）

**目标**：pipeline 收口点能产出符合 schema 的 Tier 1 候选包；resolver 读路径上线（混合检索 + 上下文预算）；本地在线消解 PoC 验证完成。

| ID | 任务 | 交付物 | AC |
|---|---|---|---|
| P1-1 | `hmopt sediment` CLI | `src/hmopt/cli/sediment.py` + Typer 注册 | 在 pipeline 末调用；遍历 `.opencode/local/runs/<run_id>/`，提取 bench delta + idea ledger 变更 + 收口的 design 摘要 → 输出 `local/sediment_staging/<run_id>.jsonl` |
| P1-2 | 蒸馏规则映射 | `src/hmopt/sediment/extractors.py`（bench→facts、review→anti_patterns、ledger→idea record）| 每类输入对应一种 extractor；单测覆盖 |
| P1-3 | 收口钩子接入 | 改 `os-opt-manager` decision 阶段 + `iterative-optimization` pass 末 + primary-agent "done" | 三处自动调用 `hmopt sediment`；不阻塞主流程 |
| P1-4 | memory export | `tools/memory_export.py`（一次性脚本）| 把现有 `memory/`、`plans/`、`reviews/` 转标准对象；产物可被 lint 通过 |
| P1-5 | 沉淀 PR 工具 | `hmopt sediment --bundle --open-pr` | 把符合晋升触发条件的候选打包成 hub PR；走 GitHub API；本仓→hub 仓 |
| **P1-6** | **resolver 读路径** | `src/hmopt/resolver/resolver.py` + 单测 | 入参 `(target, stage)`，按设计 §12.2 顺序解析：hub.skills selector 命中 → 拉 requires → 调 retrieve 查 hub.knowledge + local.memory → 合并去重 → 按 stage 预算裁切。pipeline 各阶段实际调用 |
| **P1-7** | **混合检索 + scalar 过滤** | `src/hmopt/resolver/retrieval.py`（faiss + sqlite-fts5 起步）+ `tools/build_index.py` | 实现设计 §12.1 伪代码：scalar 预过滤 → BM25 + vector RRF 融合 + entity bonus + `score` 加权；返回 top-k；retrieval.jsonl 落盘（§12.4 可观测） |
| **P1-8** | **本地在线消解 PoC**（评审反馈 ③）| `src/hmopt/memory/local_curator.py`（七路分类器，参 mem0 v0.1.x 论文版 prompt 扩展）+ 分类 benchmark | 三选一依赖决策（§17 议题 6）。**benchmark 必覆盖七路**：duplicate / contradiction / **temporal**（曾对现过时）/ **conditional**（都对、`applies_when` 不同）/ **selector**（rebase 后路径变）/ **evidence**（同 delta 不同 `compare_level`）/ novel，每类 ≥ 5 条共 ≥ 40 条。**AC**：① 七路分类准确率 ≥ 0.85；② **temporal + conditional 子类 false-delete rate ≈ 0**（误把历史/条件事实删掉是 PoC 否决项）；③ 端到端 ≤ 3s/条；④ 本地**不**跑 subsumption（留中央 P2-9） |
| **P1-9** | **LLM 显著性抽取 pass** | 扩 `extractors.py` 加 LLM pass | 规则抽取剩余的 free-form 文本（design 摘要 / reviewer 笔记）→ 跑 `FACT_RETRIEVAL_PROMPT` 风格抽取；产物默认 `confidence: tentative`；可通过 `--no-llm-extract` 关闭 |
| **P1-10** | **retrieval eval 硬门**（评审反馈 ②）| `eval/retrieval/queries.yaml` + `tools/run_retrieval_eval.py` + CI 接入 | ① **三类 query 各 ≥ 8 条**：target-anchored（`mm/vmscan.c::shrink_node`）/ mechanism-anchored（`hoist-loop-invariant`）/ free-form（"最近哪些方案被判 bad plan"）；② 每条 expected ID 标 **must-hit / optional-hit**，分别算严格 recall 与宽松 recall；③ **检索逻辑 PR 必跑**，早期用**回归门**（不劣于上次 green）、语料够大后上绝对线（must-hit recall@5 ≥ 0.8）；④ 对**符号名 query** 单独报 BM25-only / vector-only / hybrid 三 ablation，证明 hybrid ≥ 各单路 |

**DoD**：
- **P0.5-2 gate 已过**（schema 收敛 + 路径 scope 校验绿）——否则不进 Phase 1。
- 现网跑一次完整 pipeline，自动落出 ≥ 1 个合 schema 的 Tier 1 候选包；`hmopt sediment --bundle` 能产生一份本地 PR diff（不必真提）。
- **resolver 在 pipeline 各阶段实际被调用**，retrieval.jsonl 真实落盘；retrieval 硬门接入 CI，符号名 query 的 hybrid ablation 报告产出且 hybrid ≥ 各单路；must-hit recall@5 基线已记录（回归门生效）。
- **本地在线消解 PoC** 跑过：七路分类准确率 ≥ 0.85、**temporal+conditional false-delete ≈ 0**；mem0 依赖决策（议题 6）写入设计文档。

---

## 3. Phase 2 — 策展 + 合并（3-6w）

**目标**：知识合并上线（引擎 A），CI 强校验，policies 落地。

| ID | 任务 | 交付物 | AC |
|---|---|---|---|
| P2-1 | Curator-agent 提示词 | `hm-skill-hub/tools/merge_curator.md` | OpenCode 可加载；输入候选 + 现有 hub knowledge，输出去重 / 冲突 / 消解决策 |
| P2-2 | 去重器 | `tools/dedup.py` | embedding 相似度（faiss 本地）+ alias 命中；阈值可调；输出"合并/新建/冲突"三态 |
| P2-3 | 冲突消解 | `tools/conflict_resolve.py` | 同 (target, mechanism) 断言相反 → Zep 双时态：旧条目标 `superseded`、`valid_until=now`，新条目 `supersedes=[old.id]` |
| P2-4 | CI: secret-scan + lint + dedup | 扩 `.github/workflows/ci.yml` | gitleaks/trufflehog + lint + dedup 全过才允许 merge |
| P2-5 | 沉淀 PR 模板 | `hm-skill-hub/.github/PULL_REQUEST_TEMPLATE.md` | 强制列：候选来源、引擎归类、双评审 checklist |
| P2-6 | policies 增强 | promotion/merge/deprecation 文档增加"实际命令"段 | 评审人可直接执行 |
| P2-7 | 双评审配置 | `CODEOWNERS` + GitHub branch protection rules（文档） | `skills/core/` 需 owner + 流程评审 |
| **P2-8** | **晋升候选自动检测器** | `tools/promotion_detector.py`（按设计 §11.5）| 两路输入：(a) hub knowledge 聚类（mechanism + scope）簇内 `confirmations` ≥ 3 且跨 ≥ 2 contributors；(b) **P2-9 喂入**的 `subsumes[] ≥ 2` 泛化记录 → 调 LLM 蒸馏「招式 + 适用条件 + 证据（含被包含实例）」→ 自动开 `promote-candidate` PR。**纪律**：只提建议、绝不自动 merge |
| **P2-9** | **subsumption 检测器**（评审反馈 ④）| 扩 `merge_curator` + `tools/subsumption.py` | 在中央批量合并中加第三类判定：incoming vs hub 最近 k 条做 **LLM 蕴含判定**，识别「泛化包含」（B 概括 A）→ 建链 `A.subsumed_by/B.subsumes`、A 进 B 的 `source[]`（**不去重吞 A**）、emit 晋升信号。**AC**：对 mock 集（含「shrink_node hoist sc->priority」vs「reclaim 热循环 hoist loop-invariant」这类）正确判为 subsumption 而非 dup/contradiction；**≥ 2 实例**才向 P2-8 emit；单实例只建链不晋升 |

**DoD**：跑一次 PR 全流程——成员本地沉淀 → 自动提 PR → CI 全过 → Curator 标注合并方案（含 subsumption 判定）→ 双评审签字 → merge → hub 多了 ≥ 1 条 L2 knowledge 记录；**晋升检测器**对 mock knowledge 集（聚类路 + subsumption 路各一）能识别出 ≥ 1 个合理候选并开 PR，且被包含的具体实例在 PR 里作为 evidence 保留、未被删除。

---

## 4. Phase 3 — eval 门（6-10w）★长杆

**目标**：技能修改安全反喂（引擎 B），SkillOpt 半自动闭环。

| ID | 任务 | 交付物 | AC |
|---|---|---|---|
| P3-1 | 评测样本采集 | `eval/task_suites/<suite>/cases/*.yaml` | 每条 case：input target + 期望优化方向 + grading rubric；初始 ≥ 20 case 覆盖 mm/wq/hyperhold |
| P3-2 | 评测执行器 | `tools/run_evals.py` | 给定 skill 版本 + task suite，跑全 case，出 `scorecards/<skill>__<semver>.json` |
| P3-3 | 代理指标 | 静态指令数估计器 + 小样本真机 A/B 接口 | Phase 3 早期用代理；后期真机加密 |
| P3-4 | eval-gate CI | 扩 `.github/workflows/ci.yml` | 任何 `skills/**/` 变更触发 evaluator；`metrics.pass_rate` 不增即拒 |
| P3-5 | bounded edit 优化器 | `tools/skill_optimizer.py` | 输入 rollout traces + 当前 skill，输出有界 add/del/replace 编辑候选 |
| P3-6 | Pareto 前沿 | `tools/pareto.py` | per-instance score 维护；保留互补候选到 `skills/<name>/candidates/` |
| P3-7 | bad_edits 缓冲 | `skills/<name>/bad_edits.jsonl` | 被 eval 否决的编辑入库；优化器下次直接跳过 |

**DoD**：手动触发优化作业 → 优化器对一个 core skill 提出有界编辑 → eval-gate 自动跑 → 严格变好则自动开 PR；不变好则编辑入 bad_edits 缓冲；产生一份 scorecard。

**长杆原因**：内核优化 ground truth = 真机 A/B 指令数 delta，慢/贵/噪声大。Phase 3 早期必须用代理指标起步。

---

## 5. Phase 4 — 自动优化（10w+）

**目标**：闭环自动迭代日常运行；每周小版本 / 每月稳定版。

| ID | 任务 | 交付物 | AC |
|---|---|---|---|
| P4-1 | 定时优化作业 | `.github/workflows/nightly.yml`（hub 内）| nightly 跑 Collect→Normalize→Cluster→Optimize→Validate→Promote→Broadcast |
| P4-2 | 发布工具 | `tools/release.py` | 自动算 semver bump（patch/minor/major）+ 打 tag + 生成 release notes + 更新 `registry.yaml` |
| P4-3 | broadcast | `tools/broadcast.py` | 发布后自动开 PR 到业务仓更新 `skill-memory.lock` |
| P4-4 | 监控面板 | `eval/scorecards/_dashboard.md`（GitHub 渲染）| 每技能 score 趋势可视化 |
| P4-5 | 半自动→全自动闸门 | `policies/auto_merge_policy.md` | 信任阈值（连续 N 次 eval 提升 + 0 回滚）后允许自动 merge；之前必须人工 |

**DoD**：一个完整自然周内，hub 自动跑出 ≥ 1 个 patch 版本，被业务仓自动 pin 后 pipeline 行为有可测的正向变化。

---

## 6. 横切关注（贯穿所有 Phase）

| 关注 | 措施 |
|---|---|
| **安全** | 脱敏门（`redact.py`）+ CI secret-scan（gitleaks）+ CODEOWNERS；任何成员都不能直接 push 到 main，只能通过 PR |
| **性能** | lint 全量跑 < 30s（大仓需要 incremental lint）；CI eval 跑 < 30min（用代理指标 + 缓存） |
| **路径兼容** | Phase 0.5 用 symlink 兜底；Phase 1+ 在 `resolver.py` 内统一解析 |
| **文档** | 每个 Phase 同步更新 `Team_Skill_Hub_Design_CN.md` 修订行；本计划文档独立维护，每 Phase 完成后打勾 |
| **回滚** | 每 Phase 入口先打 git tag；每个 hub 发布带 scorecard，方便诊断回滚 |

---

## 7. 角色与责任（RACI lite）

| 角色 | Phase 0–2 | Phase 3 | Phase 4 |
|---|---|---|---|
| **平台 / 工具** | hub 骨架 + 工具链（R）| eval 执行器（R）| 定时作业 + 发布（R）|
| **领域专家** | 评审示例（C）| eval case 设计（R）| 评审异常（C）|
| **流程 reviewer** | policies 评审（A）| eval-gate 设计（A）| 自动 merge 闸门（A）|
| **业务仓使用者** | 用现有 .opencode 不变（I）| Phase 3.5 切到 hub-backed（C）| 消费新版本（I）|

R=Responsible, A=Accountable, C=Consulted, I=Informed。

---

## 8. 关键路径

```
P0-3(schemas) → P0-6(parser/lint) → P0-7(CI)        ← Phase 0 主链
                       ↓
                P0.5-2(schema/md 收敛) ★前置阻塞 gate ← 不过则 Phase 1 不开工
                       ↓
P1-1(sediment) → P1-2(extractors) → P1-9(LLM 显著性)               ┐
                       ↓                                              ├→ P1-6(resolver) ← 读路径主链
                P1-4(memory export) → P1-7(混合检索) → P1-10(retr 硬门) ┘
                       ↓
                P1-8(本地七路消解 PoC) → 议题 6 决策
                       ↓
P2-2(dedup) → P2-3(conflict) → P2-9(subsumption) → P2-8(晋升检测) → P3-2(evaluator) → P3-4(eval-gate) ← 长杆终点
                                                                                  ↓
                                                                              P4-1(nightly)
```

**最关键单点**：
- **P0-3**（schemas）—— 后续所有 lint / 校验 / 合并都靠它。Phase 0 已落地。
- **P1-6 + P1-7**（resolver + 混合检索）—— 读路径上线，决定了 mem0 / EverOS 的延迟与成本红利能不能拿到；任何下游 pipeline 加载都靠它。
- **P1-8**（本地在线消解 PoC）—— 决定 mem0 依赖策略，影响后续 P2 中央层 Curator 工作量分配。

---

## 9. 本会话交付清单（Phase 0 实际产出）

完成后将存在的文件：

```
hm-skill-hub/
  README.md  CONTRIBUTING.md  GOVERNANCE.md  CHANGELOG.md
  registry.yaml  .gitignore
  schemas/{bad_plan,global_lesson,memory_item,idea,
           skill_frontmatter,skill_patch,scorecard}.schema.json   # 7 份
  _registry/{mechanisms,subsystem_selectors}.yaml
  policies/{promotion,merge,deprecation}_policy.md
  tools/{parse_memory,lint,redact}.py  tools/requirements.txt
  .github/workflows/ci.yml
  skills/{core,technique,domain}/.gitkeep
  skills/core/example/SKILL.md
  knowledge/global/{lessons,anti_patterns}/{.gitkeep, H001-*.md, A001-*.md}
  knowledge/{subsystems,targets,index}/.gitkeep
  evidence/{benchmarks,regressions}/.gitkeep
  eval/{task_suites,scorecards}/.gitkeep
  staging/.gitkeep  releases/.gitkeep
.opencode/
  skill-memory.lock                                                # 占位
docs/
  Skill_Hub_Implementation_Plan_CN.md                              # 本文档
```

**验证**：`python hm-skill-hub/tools/lint.py` 在示例上 exit 0、schemas 互引无误、目录树与设计 §6.1 一致。

---

## 10. 本会话不做的事（避免范围蔓延）

- 不写 Curator-agent 提示词（Phase 2）
- 不写 SkillOpt 优化器、Pareto 算法、eval 执行器（Phase 3）
- 不真正搬动 `.opencode/{skills,agents,...}` 内容（Phase 0.5 独立会话）
- 不创建独立 GitHub 仓（受环境限制，本会话仅准备好"将来一行 subtree 拆出"的结构）
- 不接入 sediment CLI / memory export（Phase 1）
- 不写 nightly 优化作业（Phase 4）
