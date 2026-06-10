# Team Skill Hub 会话级完整审阅报告（设计 · 方案 · 实现代码）

| 项 | 值 |
|---|---|
| 审阅日期 | 2026-06-10 |
| 审阅对象 | 会话提交 `8e52a4f..307f8ee`（15 commits）：3 份设计文档（CN+EN）· `hm-skill-hub/`（20 个工具 + 82 测试 + schemas/policies/CI）· `src/hmopt/{skillhub,sediment,memory}`（11 模块 + 61 测试 + CLI 接线） |
| 审阅方式 | 3 个并行独立审阅 agent（Fable 5 模型，只读模式）分别覆盖设计 / Hub 侧实现 / 消费侧实现；全部门禁、测试与基准数字独立复跑复现；含主动 break-it 探针验证 |
| 总体结论 | **设计 B+ · Hub 侧实现 B+ · 消费侧实现 B−（库 A− / 集成 C）→ 会话总体 B+** |

---

## 0. TL;DR

这是一套**架构正确、纪律严明、披露诚实**的 PoC 级工程：双引擎分治 + 七路合并分类 + 不删铁律的设计是对 mem0 四路 ADD/UPDATE/DELETE/NOOP 的真实进步；143 个测试全绿且关键安全属性（单调 eval 门、CRDT 不删、dry-run 默认、信任阈值）被测试钉住；proxy 循环性、eval-gate 威胁模型、stub 标注等披露纪律被三位审阅者一致点名表扬。

但有一个**总头条发现**和一组**真实缺陷**必须直说：

1. **闭环尚未闭合**——`resolver` / `sediment` 是"库 + CLI + 测试"，生产 pipeline（LangGraph `graph.py::_report` 收口、OpenCode harness）**今天没有任何代码路径调用它们**。P1-3 计划状态注记已诚实承认未接线，但 P1-6 的 DoD（"resolver 在 pipeline 各阶段实际被调用"）**未达成**。当前一次优化运行从这套代码中获得的收益为零，除非人工跑 `hmopt resolve/sediment`。
2. **受信写路径有 2 个真实缺陷**（探针证实）：`nightly --apply` 在 lint/secret 门判定**之前**就改写技能文件；`conflict_resolve.apply_to_files` 只写 loser 墓碑、**丢弃 winner 的 `supersedes[]`**，双时态审计链只落了一半。
3. **双树漂移已经发生**——架构最担心的事已成现实：hub 侧 `polarity()` 的 "do: avoid X" 修复**没有回迁**到消费侧 `local_curator`，同一对记录本地判 duplicate、中央判 contradiction。8 个双树复制面中只有 1 个有 parity 测试。
4. **设计 §7 描述了一个不存在的世界**——文档把 `memory_item` 当作"唯一知识 schema"，实际是四个字段集互斥的记录家族；`valid_from/valid_until` 双时态字段只存在于 1/4 家族，跨家族 supersede 被 schema 阻断（`conflict_resolve.py` 注释里自己承认在绕）。

以上均**可在数天内修复且不动架构**——这正是 B+ 而非更低的原因。

---

## 1. 评分汇总

| 切面 | 评分 | 一句话判语 |
|---|---|---|
| 设计（3 份文档） | **B+** | 脊柱正确、研究映射诚实、v2.3 确实吸收了评审；但 §7 数据模型与实物脱节、晋升漏斗小团队会饿死、§6.1 残留矛盾树 |
| Hub 侧实现 | **B+** | 82/82 测试绿、安全属性真实落地、披露堪称范本；但受信写路径 2 缺陷 + 2 个死 schema + policy/工具阈值漂移 |
| 消费侧实现 | **B−** | 作为库是 A−（干净、确定性、跨树 schema 契约测试是全场最佳）；作为"集成"是 C（pipeline 零调用、lock 文件是死配置、极性已漂移） |

---

## 2. 设计审阅（B+ — 有条件通过，需 v2.4 修订）

### 2.1 确认的优点

1. **双资产/双引擎分治（§4.1）正确且论证充分**——"知识走集合并+冲突消解、技能走 eval 门竞争式编辑+Pareto"是全案最重要的决定；本地/中央两级引擎 A（本地不跑 subsumption）体现了真实的成本意识。
2. **七路关系分类 + 每路"绝不"列（§10.1.0）**——把 temporal/conditional 从 contradiction 中区分出来并禁止 delete，直击经典记忆系统失败模式（激进去重毁掉历史）；subsumption 作为 knowledge→technique 毕业通道的引擎是优雅的统一。
3. **markdown 真相源 + 路径编码 scope + CI 一致性强校验**（已实现并测试）——保持系统 git 原生、可评审、后端可换。
4. **诚实的长杆识别**——§15 点名 eval 套件为关键风险；Phase 3 状态注记明确否认 ProxyScorer 是真机证明。
5. **纵深防御的治理闭环**——三门 + 双评审 + 无豁免 + ≥2 实例防伪泛化 + bad_edits + 文本学习率 + 信任阈值，每个机制都能溯源到一个具名失败模式。

### 2.2 关键发现（按严重度）

| 级别 | §ref | 发现 |
|---|---|---|
| **CRITICAL** | §7 vs 实际 schemas | §7 把 `memory_item` 当唯一知识 schema 并要求"全字段、禁自扩"；实际四家族（memory_item/global_lesson/bad_plan/idea）字段集互斥：global_lesson/bad_plan 的 `superseded_by` 是**同前缀标量**（跨家族 supersede 被 schema 禁止）、无 `valid_from/until`——"双时态、永不删"的头条纪律只对 1/4 家族成立 |
| MAJOR | §6.1 | v2.3 "收敛"只完成一半：§6.1 内部仍有两棵互相矛盾的目录树（L171–181 旧多记录树 vs L190–197 新树）；plan Phase-0 文本未重新同步（`lessons/` vs `heuristics/` 三处残留） |
| MAJOR | §8/§9/§11.5 | **L1→L2 确认 livelock**：hub 检索下限 L2，staging 的 L1 候选任何人都检索不到，confirmations 只能靠独立重发现累积——3–5 人小团队冷启动期漏斗必然饿死；无 bootstrap 模式 |
| MAJOR | §6.2/§12.1 | **idea_ledger 对读路径不可见**（实测）：scalar 过滤 `status="active"`，而 idea 的状态机是 approved/landed/...——L001（landed）在真实 hub 上检索不出 |
| MAJOR | §10.1.0 | 七路分类**无互斥性、无优先级序**：kernel 版本变化同时是 temporal 和 conditional（走 temporal 会丢失旧 kernel 消费者的信息）；缺 partial-overlap / refinement / retraction 等关系 |
| MAJOR | §10.1.0 分层表 vs 实现 | 设计称 temporal/evidence 两级都跑；中央实现实际把它们路由为 additive（陈旧记录永远 active 并被继续检索）——Phase 2 状态注记未披露此偏差 |
| MAJOR | 全文 | **跨成员 ID 分配方案缺失**：F/H/A/B/V/L 全部本地顺序铸造，两个成员必然撞号；设计要求"稳定 ID"但从未说明谁分配、怎么消解——这会卡死第一个真实双人沉淀 PR |
| MAJOR | plan P1-8/P1-10 AC | **自评基准作为验收标准**：P1-8 的 1.0 准确率是同会话自建 48 例基准上的确定性分类器（构造性满足）；P1-10 的 1.0 recall 跑在合成语料上（distractor 为证明论点而造）。回归门框架是诚实的，但头条数字不会在真实数据/真实 embedder 下存活 |
| MAJOR | §9/§12.1 | score 公式无权重值、无尺度、**无生产者**——没有任何组件写 `score`，检索的 `sigmoid(score)` 当前是恒 0.5 的 no-op |
| MINOR | §4.2 | Tier×L0–L3 正交性言过其实（L0/L1 实际由 tier 位置定义，仅 Tier2 内 L2/L3 有区分意义） |
| MINOR | §6.2/§11.5 | 毕业后语义未定义：pattern B 毕业成 technique 后知识侧 B 仍 active → 双份服务 + SkillOpt 改技能后漂移；无 de-graduation 路径 |

### 2.3 §15 缺失风险（设计审阅补充的 10 项）

评审吞吐/幽灵 CODEOWNERS（5 个团队句柄不存在）· toy-embedder→真实 embedder 迁移使全部阈值与基线同时失效（无重校准任务）· schema 自身演化无版本字段/迁移工具（`additionalProperties:false` 下任何加字段都是破坏性变更）· 拆仓后 evidence 相对路径悬空 · **知识记录提示注入/投毒**（记录正文进 agent 上下文 + 未来 LLM judge 消费候选文本；P4-5 信任门会随时间侵蚀双评审缓解）· LLM 非确定性进 CI 门无缓存/重放策略 · kernel 代码片段进 hub 的 IP/保密面（redact 只盯密钥）· 跨成员索引漂移 · 自研 24 个工具的 bus factor。

### 2.4 §17 议题处置

8 个议题中：#6（mem0 策略）**处置堪称范本**（决策+理由+工件+重评触发条件全部回写）；#3（technique 层）、#4（检索后端——实际选了菜单外的第四方案）、#7（路径粒度）已被实现**事实决定但未回写**，§17 文本已陈旧；#2（eval ground truth）de-facto 选了 proxy-first 但真正决定工期的真机选项仍悬空；#1/#5/#8 真实开放。

---

## 3. Hub 侧实现审阅（B+ — 通过，附必须跟进项）

### 3.1 交付物对照矩阵（vs 计划任务 AC，从严判定）

| 任务 | 判定 | 依据 |
|---|---|---|
| P0.5-2 schema 收敛 gate | **REAL** | 一记录一文件、路径↔scope 精准报错（有测试）、lint exit 0 |
| P2-1 Curator 提示词 | PARTIAL | `merge_curator.md` 存在但 AC 说"OpenCode 可加载"——`.opencode/` 下无任何引用；背后的确定性引擎是真的 |
| P2-2 dedup | REAL | 三态 + CI `--check`；阈值 0.82 与 policy 的 0.92 漂移 |
| P2-3 conflict_resolve | **PARTIAL** | 墓碑真实、家族感知、不删；但 `apply_to_files` **从不写 winner 的 `supersedes[]`**（探针证实）——计划 AC 要求的正向边在文件路径上被丢弃 |
| P2-4 CI | PARTIAL | lint+redact+dedup+eval-gate 已接；**gitleaks/trufflehog 缺失**；hub 内 workflow 拆仓前休眠 |
| P2-5/6/7 模板/政策/CODEOWNERS | REAL（文档级） | 团队句柄是占位符 |
| P2-8 晋升检测器 | PARTIAL | 双路输入真实有测试；但无 LLM 蒸馏（机械拼接正文）、**不真正开 PR**（`--pr-body` 仅打印） |
| P2-9 subsumption | PARTIAL | 建链/≥2 实例/先于 dedup/极性守卫全有测试；但"LLM 蕴含判定"是 heuristic-only，`llm=` 钩子无任何调用方接线，judge 异常**静默回退** |
| P3-1 eval 套件 | **PARTIAL** | 9 case vs AC ≥20；且 9 个坍缩成 **3 组同质 rubric**（每子系统 3 个 case 共享 expected_terms）——套件实际只有 3 个自由度，一条 bullet 翻转 3 个 case |
| P3-2 run_evals | REAL* | 跑通出卡；但产出的 scorecard **不符合自家 `scorecard.schema.json`**（缺 `run_at`、字段名不一致、additionalProperties:false 全拒）——死 schema |
| P3-3 代理指标 | STUB 倾向 | 仅关键词覆盖 proxy；静态指令数估计器与真机接口只有 Protocol 可插拔点；三处诚实披露 |
| P3-4 eval_gate + CI | REAL | 在根 CI 真实运行；in-tree 可变基线的共改绕过已在 docstring 明示 |
| P3-5 优化器 | PARTIAL | 有界编辑/严格变好/bad_edits/版本累积全真实；但只实现 `op=add`（del/replace 抛 NotImplementedError）vs 计划的 add/del/replace |
| P3-6 Pareto / P3-7 bad_edits | REAL | 正确且有测试钉住 |
| P4-1 nightly | **PARTIAL** | 7 步编排、dry-run 默认、hub_root 参数化均真；但见下方时序缺陷 |
| P4-2 release | PARTIAL | bump 推断/registry/notes 真实；**无任何 git tag**（docstring 推给 CI，CI 无 tag 步骤）；无变更时仍铸 patch 版本 |
| P4-3 broadcast | PARTIAL | lock 再生成真实；`--open-pr` 显式 print-stub |
| P4-4 dashboard / P4-5 信任门 | REAL / PARTIAL | 信任计算真实且与 policy 一致；但判定**纯咨询**——没有任何东西消费 `auto` vs `human` 去改变 merge 行为 |

### 3.2 关键发现

| 级别 | 位置 | 发现 |
|---|---|---|
| **MAJOR** | `nightly.py:103-116 vs :147-153` | `--apply` 下第 (4) 步 `optimize(apply=True)` 在 normalize/validate 门判定**之前**执行；只有 Promote/Broadcast/Dashboard 被门控。探针证实：secret-scan 失败的 hub 仍被改写 best_skill.md + SKILL.md bump + 新 scorecard（exit 1 但变更落盘）|
| **MAJOR** | `conflict_resolve.py:104-110` | `apply_to_files` 丢弃已计算的 `winner_fields["supersedes"]`——审计链半写，winner 文件随后会过不了流程评审清单的"双时态字段完整" |
| **MAJOR** | `schemas/{scorecard,skill_patch}.schema.json` | **死契约**：仓内唯一真实 scorecard 不过自家 schema；没有工具校验 scorecard 或产出 skill_patch manifest，而 PR 模板 Gate 2 又要求它——schema 和产出方今天同时在船上互相矛盾 |
| MEDIUM | `hub_records.py:213-215` | `load_hub_knowledge` 对解析失败 `except Exception: continue`——所有引擎 A 门禁（dedup --check/curate/promotion/nightly）在一个**静默缩小的 hub**上推理；CI 门的 fail-open |
| MEDIUM | `dedup.py:52-54` | `_related` 两个分支都要求 `alias_hit`——**无 mechanism 的记录（全部 H/V lesson）永远无法被判 merge 或 conflict**，相反结论的 heuristic 冲突门检测不到 |
| MEDIUM | policy vs tool | dedup 阈值 `merge_policy.md` 写 0.92、`dedup.py` 默认 0.82——评审人按 policy 预期的门比 CI 实际执行的更严 |
| MEDIUM | staging 流 | `staging/*.jsonl` 候选**从不过 schema 校验**（lint 只读 knowledge/+skills/ 的 .md）——promotion policy 的 Gate 1 对 L1 形态未执行 |
| MEDIUM | 根 CI | hub 内 ci.yml/nightly.yml 拆仓前休眠（已注明）；存活的根 workflow **在 feature 分支 push 时不触发**（本会话全部提交未经 CI 检查除非开 PR）；ruff 不在任何 CI 里 |
| MINOR | `redact.py` | `generic-hex-key {40,}` 恰好命中 40 位 git SHA（与"evidence 必须可解析 commit hash"的政策互踩）；`[REDACTED]/[FAKE]` 行级 allow-tag 是单 token 自我旁路；无 `--check` 时有命中也 exit 0（与自家 docstring 矛盾）|
| MINOR | `conflict_resolve.py:90-97` | `yaml.safe_dump` 全量重写 frontmatter 风格有损：一次语义无操作重写产生 **37 行 diff**（引号/流式列表/折行全变）——对以审计为中心的仓库，一字段变更被淹没在噪声里 |
| MINOR | `_registry/mechanisms.yaml:3` | 注释声称"CI 拒未注册 mechanism"——没有任何工具做此校验；愿景注释当事实发货 |

### 3.3 横向质量（跨工具视角）

- **重复/漂移清单**：frontmatter 正则 7+2 处（3 个语义变体，EOF 容忍度已漂移）；`_semver` 4+1 处；`sys.path.insert` 样板 16 处；maturity rank map 5 处；token-hash embedder hub↔consumer 字节等价但**无同步校验**。结论：`tools/_common.py` 是正当的（similarity/hub_records 已确立 tools 内共享模式），可消 ~150 行重复。
- **CLI 一致性**：15 个 CLI 三种解析风格混用；除 dedup 外全部**静默忽略空格形式**（`release.py --bump major` 静默跑 auto）——是 foot-gun 不是风格问题。
- **错误处理**：门禁 fail-closed、报表 degrade 的分界大体正确，例外正是 `load_hub_knowledge`（门禁输入却 fail-open）；dedup/central_curate 对坏输入裸 traceback（rc 1，安全但违反自家 usage=2 约定）。
- **测试质量**：最有承重力的 5 个——单实例回归必须拒绝（THE 安全属性）、三 schema 家族 superseded 后重 lint、真实 hub subsumption 恰为 {(H001,F001)} **且显式断言相似度余量**（阈值侵蚀金丝雀，罕见且优秀）、版本 bump+scorecard 累积、nightly dry-run 零变更+temp-hub 往返。最弱的 5 个——`glob()` 真值断言（恒真）、`bump_version` 同构重述等 4 个套套逻辑。
- **安全**：全仓 `yaml.safe_load`、无 eval/exec/subprocess；未来 LLM judge 的提示注入面已随代码提交（`_entailment_prompt` 嵌入攻击者可控文本 + 子串解析 + 静默回退）；恶意记录可自带 `subsumes:[..]` 直喂 promotion 路径 1（人工门控故影响为评审噪声，但检测器信任未验证的自断言链接）。

---

## 4. 消费侧实现审阅（B− — 库 A−，集成 C）

### 4.1 集成真实度矩阵

| 计划项 | 判定 | 现实 |
|---|---|---|
| P1-1 `hmopt sediment` CLI | **CLI-ONLY** | 命令存在且工作；**无 pipeline 调用** |
| P1-2 extractors | WIRED（入 CLI 路径） | 完整、按真实 hub schema 校验（全场最佳测试）；从未被真实 run 喂过 |
| P1-3 收口钩子 | **SCAFFOLD（未开始）** | `graph.py::_report`(1181-1207) 无调用；`grep -r sediment .opencode/` 零命中；状态注记已承认 |
| P1-4 memory export | **ABSENT** | `tools/memory_export.py` 不存在——且状态注记**未标注此缺失**（唯一未披露的缺口） |
| P1-5 --bundle/--open-pr | --bundle CLI-ONLY；--open-pr ABSENT | 已承认 |
| P1-6 resolver | **CLI-ONLY，DoD 明确未达成** | 库端到端正确（实测挂载 F001/守预算）；唯一调用方是 CLI+测试，DoD 说"被各阶段实际调用" |
| P1-7 混合检索 + build_index | 检索 WIRED（库）；索引 SCAFFOLD | §12.1 栈忠实实现；`build_index.py` 不存在、**无持久索引**——每次 `Resolver()` 全量重嵌（§12.3 明令禁止全量重建为常规路径） |
| P1-8 本地 curator PoC | SCAFFOLD（诚实 PoC） | 四项 AC 全过；但 `apply_relation` 返回动作字符串——没有任何代码把真实记录转成 `CuratorItem` 或改写内存文件 |
| P1-9 LLM 显著性 | CLI-ONLY | 离线 no-op 正确；真实 LLM 路径从未行使 |
| P1-10 retrieval 硬门 | **WIRED（CI via pytest）** | 绝对线+ablation+基线回归都在 CI 跑；瑕疵：CLI 路径只查基线回归不查绝对 0.8 线（与模块 docstring 不符） |

**直白总结**：读路径与写路径是真实、有测试的库；*闭环*——pipeline 先读知识再规划、决策后沉淀——只存在于设计文档。`opencode/pipeline.py::_infer_memory_paths` 仍只注入旧版扁平文件。

### 4.2 关键发现

| 级别 | 位置 | 发现 |
|---|---|---|
| **MAJOR** | `resolver.py:218-243` | **跨语料分数合并未校准**：hub 与 local 各自独立的 RRF 排名分直接混合。实测：1 条半成品 local L1 笔记与策展 hub L2 F001 **得分完全相同（0.1164）**——两者之间的排序是 dict 迭代运气；maturity 只影响预过滤和裁切、从不影响排名 |
| **MAJOR** | `local_curator.py:64-79` vs `hub_records.py:145-164` | **极性逻辑已经漂移**：hub 先扫负词（"do: avoid X"→−1），消费侧先查 `do:` 前缀（→+1）——hub 的 review 修复未回迁。同一记录对本地/中央分类可不一致，正是设计要防的引擎内分歧 |
| MEDIUM | `resolver.py:179-180` vs §12.4 | hub 缺失→裸 `FileNotFoundError`，无降级快照、无警告续行；**`.opencode/skill-memory.lock` 是死配置**——没有任何代码读它的 path/pin/hub_version，retrieval.jsonl 不记录消费版本（§12.4 要求） |
| MEDIUM | `retrieval.py:170-190` | **混合模式无相关性下限**：vec_rank 取 top-pool 含零余弦文档，每条入 scope 记录都拿到 RRF 质量。实测：零词重叠记录以 0.0081 检出；真实 resolve 中 V001（单镜像测试陷阱）被挂进 shrink_node hoist 查询——正是 §12.1 引 mem0 警告的上下文污染 |
| MEDIUM | `extractors.py:99` + `pipeline.py:92-105` | **临时 ID 在 bundle 接缝碰撞**：seq 每 run 重置，每次都产 F901/A901/…；`bundle_staging` 直接拼接无重编号——一个 bundle 携带 N 条同名 F901，hub lint 的 id_sink 会在 PR 时拒掉第二条；extractors 注释称"Curator 分配最终 ID"——**没有任何工具做这件事** |
| MEDIUM | `cli.py:14-33` | 模块级导入（pydantic/langgraph/llama_index）**击穿了三个新命令内部精心的懒加载**：本沙箱 `python -m hmopt.cli --help` 直接 ModuleNotFoundError，而库本体只需 pyyaml+jsonschema 即可完整工作（已验证） |
| MEDIUM | `records.py:31` | `confidence: tentative→L1` 使**真实 hub 记录 A001 在默认 L2 下限下永不可检索**；今后所有 review→anti_pattern/salience 产物（出生即 tentative）合并后同样不可达，直到 confirmations 提升——eval 语料全部钉 L2 把这事藏住了 |
| MINOR | `local_curator.py:57` | `kernel_version: float`——6.10 解析为 6.1 < 6.9，**x.10+ 内核全部反转 temporal 判定** |
| MINOR | `retrieval.py:101-107` | 每次构造全量 tokenize+embed、每次 CLI 重建、无持久化无增量（§12.3 差距）；7 条记录无感，规模化后是 O(N·doc)/次 |
| MINOR | `extractors.py:79-82` | 第 **3** 份 `_slugify` 拷贝——parity 测试只覆盖 resolver↔pipeline 一对，extractors 这份可无声漂移并破坏 target_slug 连接 |

### 4.3 双树漂移清单（同逻辑两棵树）

8 个复制面：embedder（今日字节等价、**无 parity 测试**）· polarity（**已漂移**）· strength（已漂移：hub confirmed→4 > L3→3，消费侧无 confidence 通道——contradiction supersede 方向可在两级间翻转）· frontmatter 解析（hub 字符串化日期、消费侧不做——消费侧加载的记录过 jsonschema `format: date` 会炸）· infer_kind vs 路径 dispatch（A 前缀去向不同）· confidence→maturity 映射 · `_jaccard`（消费侧不拆 snake_case）· `_slugify` ×3。**结论**：双树复制作为拆仓就绪设计可接受，*但前提是每个复制面都有 parity 测试*——现状 8 中有 1，且 1 个已实漂。

---

## 5. 三份报告的交叉验证点

以下发现被 ≥2 位独立审阅者从不同切面命中，可信度最高：

1. **死 schema / §7 失真**（设计 CRITICAL + hub MAJOR）：设计叙事、schema 文件、工具产物三方互相矛盾。
2. **tentative/maturity 可见性**（设计 MINOR-17 + 消费侧 MEDIUM-F8）：A001 进了 Tier2 却低于检索下限，confidence↔maturity 映射是承重决策但无文档。
3. **基准自评性**（设计 MAJOR-9 + hub P3-1 判定 + 消费侧测试质量节）：1.0 全是 fixture-fitted；回归门框架诚实，绝对数字不可外推。
4. **LLM judge 注入面**（设计风险 6 + hub MINOR-17）：提示注入面已随代码提交，信任门会随时间侵蚀人工缓解。
5. **嵌入器迁移悬崖**（设计风险 2 + 双侧阈值盘点）：全部阈值/基线/ablation 结论锚定 toy embedder，换真模型须整体重校准且计划无此任务。

---

## 6. 分级修复路线图（综合三份 Top-10，去重排序）

### P0 — 在沙箱外运行 `--apply` / 提交真实沉淀 PR 之前必须修（合计 ~1 天）

1. `nightly.py --apply`：normalize 门失败时禁止 optimizer 写入（或 optimizer 先 dry-run，门过后再写）；测试改为钉住 best_skill/SKILL/scorecards 全部不变（当前只钉 registry.yaml）。
2. `conflict_resolve.apply_to_files`：写回 winner 的 `supersedes[]`（双文件），在三家族测试中断言。
3. 把 hub 的 polarity 修复回迁 `local_curator`，并为 polarity/strength/embedder 各加 parity 测试（仿 `_slugify` 模式）。

### P1 — 让闭环成真（~1-2 周）

4. **接一个真实钩子端到端**：`graph.py::_report` 内 try/except 调 `sediment_run`，和/或 manager decision 阶段提示词中写入 `hmopt sediment` 指令——在此之前，计划中 P1-3/P1-6 应改标"未开始"。
5. 跨语料排名校准（单一融合检索器带 origin 字段，或按 maturity 降权 local），加"hub L2 ≥ 平凡 local L1"测试。
6. 检索相关性下限（`bm25==0 且 cosine<ε` 即丢弃），用无关记录测试（今天会失败）。
7. 临时 ID 命名空间化（`F901@<run_id>` 或 bundle/curate 时重编号），双 run bundle 测试。
8. 调和死 schema：run_evals 产物补 `run_at`/对齐字段并由 lint 校验 scorecard；skill_patch 要么真产出要么删 schema+模板要求。
9. `load_hub_knowledge` fail-closed（收集并上抛解析失败，门禁有跳过即 fail）；staging JSONL 过 schema 校验（Gate 1 落地）。
10. CLI 顶层重导入下沉到各命令内（恢复轻量命令的独立可运行性）；repo root 用 git toplevel 而非 CWD；读取 lock 文件并在 retrieval.jsonl 记录 hub_version。

### P2 — 设计 v2.4 修订（文档，~2-3 天）

11. §7 重写为四家族模型 + 共享生命周期核心（status/superseded_by[]/subsumes/valid_from-until/contributor 跨家族必备，`superseded_by` 全部数组化并允许跨家族引用）。
12. 删除 §6.1 残留旧树；同步 plan Phase-0 文本（lessons→heuristics ×3）与 Phase 1 工期（2-3w vs 3-5w 四处不一致）。
13. 补漏斗 bootstrap 政策（L1 带 tentative 标签可检索/低权重，或单贡献者临时 L2+强制过期）；写跨成员 ID 分配方案。
14. §12.1 增加每家族"可检索状态"映射（idea: approved/landed 可见）；§15 追加 10 项缺失风险；§17 回写 #2/#3/#4/#7 的事实决定。
15. 七路分类增加判定优先级序与 contradiction/conditional 边界规则；中央层 temporal/evidence 路由要么实现要么改表。

### P3 — 工程质量（机会性）

16. `tools/_common.py` 抽取（semver/frontmatter/maturity-rank/_as_list/scorecard 收集，~150 行）；统一 CLI 解析（接受两种 flag 形式、拒绝未知 flag）。
17. eval 套件扩到 ≥20 case 且组内 rubric 差异化（当前 3 个自由度）。
18. 根 CI 增加 feature 分支 push 触发 + ruff + gitleaks；nightly 增加 git tag 步骤。
19. dedup 阈值统一（0.82 或 0.92 二选一，policy 与代码互引并加 pin 测试）；frontmatter 定向写替代全量 safe_dump（保护审计 diff）。
20. `kernel_version` 改字符串/元组比较；mechanism 注册校验落地（或删掉愿景注释）。

---

## 7. 诚实性评估（审阅者一致结论)

本会话的**披露纪律是其最大优点**：proxy↔proposer 循环性、eval-gate 共改绕过威胁模型、`--open-pr` stub、P1-3 未接线、toy embedder 的非语义本质——全部在 docstring/commit/CHANGELOG/状态注记中主动声明，三位审阅者均独立点名表扬，且多数声明在被探针验证时成立。两处披露缺口：P1-4（memory_export 缺失）未在状态注记中标注；Phase 2 状态注记暗示 §10.1.b 七路全实现（实际中央层 temporal/evidence 走 additive）。头条数字（curator 1.0、retrieval 1.0、optimizer 0.67→1.00）**全部可复现但全部是 fixture 构造性结果**——作为控制流验证成立，作为能力证明不成立，这一点本身也已被披露。

**最终判语**：架构正确、地基扎实、纪律出色的 B+ 级工程。它今天是一个"可被验证的承诺"而非"已闭合的回路"——P0 三项修复 + P1 第 4 项（接通第一个真实钩子）落地后，即可升至 A− 并开始产生真实价值。
