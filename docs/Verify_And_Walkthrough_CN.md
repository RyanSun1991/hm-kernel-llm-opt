# Team Skill Hub — 原理、流程与逐 Phase 验证手册

> 目的：用大白话讲清这套系统**是什么、怎么转、怎么设计的**，并给出**每个 Phase 可直接复制粘贴运行**的验证命令 + 预期输出 + 怎么解读。
> 本手册中的命令与输出均在本仓 `claude/admiring-allen-6j86r8` 分支上实跑验证过。

---

## 一、一句话 + 大白话

**一句话**：让"AI 优化内核代码"过程中积累的经验（哪些招式有效、哪些是坑、某函数该怎么改），能**跨成员积累、自动验证、安全复用**，形成一个**消费 → 蒸馏 → 晋升 → 发布 → 再消费**的闭环。

**大白话**：现在每个工程师用 AI 跑内核优化，经验只留在自己电脑里（`.opencode/` 本地记忆）。这套系统把这些经验**外置到一个中央仓 `hm-skill-hub`**，加上**质量门 + 版本号**，让团队像用"私有 npm 包"一样消费经验，且经验会越用越准。

---

## 二、核心设计思路（4 个关键决定）

### 决定 1：两类资产，两台引擎（最重要）

| | **Skills（技能 = 做法/流程）** | **Knowledge（知识 = 事实/教训）** |
|---|---|---|
| 例子 | "出方案前先查 bad_plans 去重"这套流程 | "shrink_node 里 sc->priority 重复读，hoist 出来省 0.8%"这条事实 |
| 怎么改 | **就地编辑**（改措辞），用 **eval 门**：留出测试集上**严格变好**才接受 | **追加**，用**去重 + 冲突消解**：绝不删历史，矛盾就双时态保留 |
| 引擎 | **B**（SkillOpt：竞争式编辑 + Pareto）| **A**（集合并 + 七路关系分类）|

> **为什么分开**：用同一套 git 行级合并治理两者 → 知识会自相矛盾、技能会被某人一周的坏经验覆盖。**两套引擎、两道门，分开走。**

### 决定 2：知识合并不是"重复?/矛盾?"二分，而是**七路关系分类**

incoming 与已有记录的关系分 7 类，**铁律：除"明确矛盾且新证据更强"外，任何分支都不物理删除**：

`duplicate`(合并出处) · `contradiction`(旧记 superseded 不删) · `temporal`(曾对现过时，保留可审计) · `conditional`(都对、条件不同，共存) · `subsumption`(B 泛化 A，A 留作 B 的证据) · `selector`(rebase 路径变，重解析) · `evidence`(同 delta 不同口径，合并)

### 决定 3：markdown 是唯一真相源，索引是派生缓存

每条知识 = 一个 `.md` 文件（YAML frontmatter + 正文）。删掉索引能从 markdown 重建；可 git 评审、可手工编辑、后端可换（faiss/pgvector 随意）。

### 决定 4：技能 = "可训练的外部参数"

把技能文本当成冻结模型的"外部参数"。改动必须在留出 eval 集上**严格变好（且零回归）**才接受；不变好的编辑进 `bad_edits` 缓冲下次跳过。这是**反喂安全**的根本机制。

---

## 三、整体流程（闭环四步）

```
   ┌──────── hm-skill-hub（中央仓·团队共享·semver）────────┐
   │   skills/（引擎B SkillOpt）   knowledge/（引擎A 合并）  │
   └────▲──────────────────────────────────────┬─────────┘
        │ ④ 发布：eval门过→升版本+tag+scorecard   │ ① 消费：pin版本 + 检索
        │                                         ▼
   ③ 晋升/合并（Curator + CI）            ┌──────────────────┐
     去重/冲突/subsumption/eval/脱敏       │ pipeline 运行(每成员) │
        ▲                                 └────────┬─────────┘
     staging/（候选）                              │ ② 蒸馏 收口点
        └──────  validated delta / 反模式 ◄────────┘
```

**各 Phase 在闭环里的位置**：

| Phase | 做什么 | 对应闭环步骤 |
|---|---|---|
| **0.5** | 把知识落盘格式收敛成"一记录一文件 + schema 校验"（地基）| 前置 |
| **1** | **① 消费**（resolver 读路径）+ **② 蒸馏**（sediment 写路径）| ①② |
| **2** | **③ 晋升/合并**（中央 Curator：dedup/conflict/subsumption/promotion）| ③ |
| **3** | 技能侧 **eval 门 + SkillOpt**（反喂安全）| ③ 的技能分支 |
| **4** | **④ 发布** + nightly 自动闭环（release/broadcast/dashboard）| ④ + 自动化 |

---

## 四、环境准备

```bash
# hub 工具 + skillhub 库只需这三个（轻量、离线、确定性）：
pip install pyyaml jsonschema pytest

# 完整 hmopt CLI（hmopt sediment/resolve/...）还需要重型栈：
pip install -e .     # 装 pydantic / langgraph / llama-index 等
```

> **注**：本手册的验证**只用前三个轻量依赖**就能跑完（hub 工具 + skillhub 库）。完整 `hmopt` CLI 命令需要重型栈，若环境没装会在顶层 import 处报 `ModuleNotFoundError: pydantic`——这不影响下面任何一条验证命令。

所有命令的工作目录：`cd` 到仓库根 `hm-kernel-llm-opt`。hub 工具命令需先 `cd hm-skill-hub`。

---

## 五、逐 Phase 验证

### Phase 0.5 — schema 收敛 gate（地基）

**原理**：每条知识一文件，frontmatter 走标准 schema，文件路径编码 scope 且与 frontmatter 必须一致（CI 强校验）。

```bash
cd hm-skill-hub

# 1) 全量 lint：schema + 路径/scope 一致性 + ID 唯一性 + scorecard 校验
python3 tools/lint.py
#   预期: OK — 6 record file(s), 4 skill(s), 1 scorecard(s) validated.   exit 0

# 2) 密钥扫描（脱敏门）
python3 tools/redact.py --check
#   预期: OK — no secret patterns matched in 1 root(s).   exit 0

# 3) 看一条记录怎么被解析成 schema object
python3 tools/parse_memory.py knowledge/targets/mm-vmscan-c-shrink-node/facts/F001-hoist-sc-priority.md
#   预期: 打印出 id/type/scope/source/... 的 JSON
```

**怎么验"门真的在拦"**：故意造一条路径说"targets/"(函数级)、frontmatter 却写 `scope.level: global` 的记录，lint 会**精准报错**：
```
F099: scope mismatch — path is targets/foo/ but scope.level='global' (expected a target-relative level ...)
```

---

### Phase 1 — 读路径（消费）+ 写路径（蒸馏）

**原理**：
- **读**：给一个 target（如 `mm/vmscan.c::shrink_node`），`resolver` 用 selector 匹配 domain skill → 拉 requires（core+technique）→ 检索挂载该函数的 knowledge → 按 stage 预算裁切。检索栈 = scalar 预过滤 → BM25 + 向量 RRF 融合 → score 加权。
- **写**：pipeline 收口时，`sediment` 把 bench delta → fact、review 否决 → anti_pattern、ledger 变更 → idea，产出合 schema 的 Tier-1 候选。

```bash
cd /home/user/hm-kernel-llm-opt   # 回仓库根（skillhub 库用 PYTHONPATH=src）

# 1) 读路径：resolver 解析一个 target
PYTHONPATH=src python3 -c "
from hmopt.skillhub.resolver import Resolver
ctx = Resolver('.').resolve('mm/vmscan.c::shrink_node', stage='research')
print('subsystem:', ctx.subsystems)
print('skills:', [s.ref for s in ctx.skills])
print('knowledge:', [(h.record.id, round(h.score,3)) for h in ctx.knowledge])
print(f'预算: {ctx.token_used}/{ctx.token_budget} tokens')
"
#   预期: subsystem ['mm-reclaim'] / skills core+technique+domain / knowledge [('F001', ...)]

# 2) 混合检索 ablation（符号名 query 证明 hybrid ≥ 单路）
PYTHONPATH=src python3 -c "
from hmopt.skillhub.records import load_records
from hmopt.skillhub.retrieval import HybridRetriever
r = HybridRetriever(load_records('hm-skill-hub/knowledge'))
for m in ('hybrid','bm25','vector'):
    print(m, '->', r.retrieve('shrink_node', mode=m, k=1)[0].record.id)
"

# 3) 本地七路 curator PoC（准确率 + false-delete 硬指标）
PYTHONPATH=src python3 -c "
from hmopt.memory.curator_benchmark import run_benchmark
b = run_benchmark()
print(f'准确率={b.accuracy:.2f} false-delete={b.false_delete_count} 本地零subsumption={b.local_subsumption_emitted==0}')
"
#   预期: 准确率=1.00 false-delete=0 本地零subsumption=True

# 4) retrieval 硬门：query 集 + ablation + 基线回归
PYTHONPATH=src python3 -c "
from hmopt.skillhub.retrieval_eval import run_eval
r = run_eval()
print(f'must-recall@5={r.must_recall:.2f} n_queries={r.n_queries}')
print('ablation:', {k:round(v,2) for k,v in r.ablation.items()})
"
#   预期: must-recall@5=1.00 n_queries=26 / ablation bm25=1.0 vector=0.8 hybrid=1.0
```

**怎么解读**：`vector=0.8 < hybrid=1.0` 是**真实**效果——纯向量在符号名（如 `shrink_node`）上拉胯，BM25 救场，所以 hybrid 不劣于任一单路。这是设计 §12.1 的核心论点被数据证明。

> **写路径**（sediment）需要构造一个"运行产物"再蒸馏，命令稍长，见 §六 一键脚本里的 sediment 段；其单测在 `tests/test_sediment.py`（18 例）。

---

### Phase 2 — 中央策展 + 合并（引擎 A）

**原理**：把一批沉淀候选与 hub 现有知识做七路分类。subsumption 先于 dedup（泛化对不能被当重复吞掉）；≥2 个不同实例才向晋升检测器 emit。

```bash
cd hm-skill-hub

# 1) subsumption：识别"泛化包含"链
python3 tools/subsumption.py
#   预期: H001 subsumes ['F001']  [link-only (1 instance)]   ← H001(启发式)泛化F001(具体事实)

# 2) DoD 端到端：一批候选跑七路决策 + 晋升信号
cat > /tmp/batch.jsonl <<'EOF'
{"schema":"memory_item","record":{"id":"F011","type":"fact","mechanism":"hoist-invariant","scope":{"level":"function","target_slug":"mm-page_alloc-rmqueue"},"status":"active","maturity":"L2","contributor":"carol","body":"in rmqueue hoist the loop-invariant zone watermark read out of the per-order loop"}}
EOF
python3 tools/central_curate.py /tmp/batch.jsonl
#   预期: F011 → subsumption (vs H001)；并 emit 晋升候选 technique/h001
#         evidence ['F001','F011'] —— 被包含的具体实例作为证据保留，不删

# 3) dedup 三态门（CI：有未消解 conflict 即 exit 1）
python3 tools/dedup.py /tmp/batch.jsonl --check; echo "exit=$?"

# 4) 冲突消解（双时态，永不删）—— 看墓碑怎么写（在三家族测试里验证）
#   见 tools/tests/test_central_curator.py::test_superseded_loser_lints_clean_for_all_schema_families
```

**怎么解读**：F011（rmqueue 的 hoist）是 H001 这条"hoist loop-invariant"启发式的第二个实例（第一个是 hub 里的 F001）→ 跨合并边界凑齐 ≥2 实例 → **自动 emit 一个"把 H001 毕业成 technique skill"的晋升候选**，且 F001/F011 作为 evidence 保留。这就是 §11.5 的 knowledge→skill 毕业通道。

---

### Phase 3 — eval 门 + SkillOpt（引擎 B，反喂安全）

**原理**：技能文本 = 可训练参数。优化器对失败的 case 组合成一条**有界编辑**，应用后跑 eval，**只有严格变好且零回归才接受**；否则进 bad_edits。`core/instruction-count-first` 的种子 `best_skill.md` **故意不完整**（缺 hoist 招式），留给优化器改进空间。

```bash
cd hm-skill-hub

# 1) 跑 eval 出 scorecard（种子 pass_rate 0.67，mm 类 case 失败）
python3 tools/run_evals.py skills/core/instruction-count-first | head -3
#   预期: pass_rate=0.67 mean=0.733 (n=9)
#   注意: 这条命令会重写 scorecard 的 run_at 时间戳。验证后用
#        git checkout -- skills/core/instruction-count-first/scorecards/ 复原

# 2) 优化器 dry-run：提一条 hoist 编辑 → 过 eval 门 → 接受
python3 tools/skill_optimizer.py skills/core/instruction-count-first | head -4
#   预期: baseline 0.67 → final 1.00；accepted 1 [hoist-invariant]；Pareto 收录

# 3) eval-gate（反喂安全门）：技能改动使 pass_rate 降即拒
python3 tools/eval_gate.py; echo "exit=$?"
#   预期: ok — pass_rate 0.67, no regression   exit 0

# 4) auto-merge 信任门（半自动默认）
python3 tools/auto_merge_gate.py
#   预期: [human] ... 0 consecutive improvement(s) < 3 required   ← 未攒够信任，必须人工 merge
```

**怎么验"安全属性真的成立"**（单调门）：`tests/`… 里有一条专门测试——一个编辑即使**提升 pass_rate 和 mean**，只要回归任一 case 就被**拒绝**（`test_safety_property_aggregate_up_but_one_instance_regresses_is_rejected`）。这是反喂安全的根本属性。

---

### Phase 4 — 自动优化闭环（nightly + 发布）

**原理**：把前面的引擎串成 nightly 作业：Collect → Normalize → Cluster → Optimize → Validate → Promote → Broadcast。**默认 dry-run 半自动**（自动提 PR、人工 merge）；`--apply` 是受信路径且按传入 hub 参数化（绝不污染真实仓）。

```bash
cd hm-skill-hub

# 1) nightly 完整闭环（dry-run，零副作用）
python3 tools/nightly.py | sed -n '/Nightly report/,/applied/p'
#   预期: normalize ok / optimize 1 accepted / validate ok / promote 升版本 / auto-merge human / applied False

# 2) release：semver 自动 bump
python3 tools/release.py
#   预期: 推断出 major/minor/patch（取决于变更）+ 理由

# 3) broadcast：生成消费端 skill-memory.lock
python3 tools/broadcast.py --hub-version=0.2.0 --sha=abc123
#   预期: 打印 mode/path/pin/hub_version 的 lock 内容

# 4) dashboard：每技能 score 趋势
python3 tools/dashboard.py | head -8

# 5) 验证 dry-run 零副作用
cd /home/user/hm-kernel-llm-opt && git status --short hm-skill-hub/ | grep -v '^??' || echo "hub 无未提交改动 ✓"
```

**怎么解读**：`promote` 那行若显示 `major / schema contract changed`，是因为 schema 文件改过（`schema_hash` 变了）→ release.py 正确识别为破坏性变更升大版本。这演示了"schema 演化检测"。

---

## 六、一键全验证（最省事）

```bash
cd /home/user/hm-kernel-llm-opt

# A) 全量测试（154 例：消费侧 61 + hub 工具 93）—— 一条命令验证全部 Phase
PYTHONPATH=src python3 -m pytest \
  tests/test_skillhub_retrieval.py tests/test_skillhub_resolver.py \
  tests/test_sediment.py tests/test_local_curator.py \
  tests/test_retrieval_eval.py tests/test_dual_tree_parity.py \
  hm-skill-hub/tools/tests/ -q
#   预期: 154 passed

# B) hub 三道门（lint + 脱敏 + eval-gate）
cd hm-skill-hub
python3 tools/lint.py && python3 tools/redact.py --check && python3 tools/eval_gate.py

# C) hub 工具也能脱离 pytest 独立跑（拆仓就绪）
python3 tools/tests/test_tools.py            # Phase 0.5 工具自测
python3 tools/tests/test_central_curator.py  # Phase 2
python3 tools/tests/test_skillopt.py         # Phase 3
python3 tools/tests/test_phase4.py           # Phase 4
```

---

## 七、能跑 / 跑不了（诚实边界）

| 能直接跑（只需 pyyaml/jsonschema/pytest）| 跑不了 / 需重型栈 / 需真机 |
|---|---|
| ✅ 全部 hub 工具（lint/dedup/subsumption/.../nightly dry-run）| ❌ 完整 `hmopt sediment/resolve` CLI（需 pydantic/langgraph）|
| ✅ skillhub 库（resolver/检索/curator/eval）直接 import | ❌ resolver/sediment **被真实 pipeline 调用**的端到端（需 langgraph 跑 graph.py）|
| ✅ 154 个单测 | ❌ 真机 A/B 指令数 delta（Phase 3 长杆；当前用关键词覆盖 proxy 代替）|
| ✅ nightly/release/broadcast/dashboard dry-run | ❌ `--apply` 真改 hub + 真开 GitHub PR（broadcast/promotion 的 `--open-pr` 是 stub）|

**重要诚实标注**：
- 所有头条数字（curator 1.0、retrieval 1.0、optimizer 0.67→1.00）**可复现但都是 fixture/proxy 构造性结果**——作为"控制流/流水线打通"的验证成立，作为"能力证明"不成立（toy 嵌入器 + 关键词代理）。真机指标是后续长杆。
- sediment 写路径已接进 `graph._report`（收口钩子，非阻塞），但端到端只能在装了 langgraph 的环境验证；本沙箱里 hook 本体有单测、调用点 py_compile 通过。
- 详细的逐项交付真实度（REAL/PARTIAL/STUB）、已知缺陷与修复进度见 `docs/Session_Implementation_Review_CN.md`。
