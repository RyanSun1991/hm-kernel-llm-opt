# 通用优化闭环：本地运行与接入指南

本指南对应 `src/hmopt/evolution/` 的第一版实现。平台以 evidence、pattern、candidate、plan、review、experiment、skill 为通用对象；内存、性能、可靠性场景通过规则、指标和执行适配器表达。

2026-09-10 V2 已新增来源分页、批量审阅、任务派发、IC 转换、正确性验收与反馈治理。首次使用请从 [V2 操作指南](EVOLUTION_V2_OPERATIONS_CN.md) 开始，以下保留基础审批与实验契约；第 9 节测试数字是第一版历史记录。

## 1. 安装和验证入口

要求 Python 3.10+、Git。可在自己的虚拟环境安装最小运行依赖，无需先启动 Neo4j、模型服务或设备服务。

PowerShell，在仓库根目录运行：

```powershell
python -m pip install -r requirements-evolution.txt
$env:PYTHONPATH = "src"
python -m hmopt.evolution.cli --help
python -m hmopt.evolution.cli --root data/evolution init
```

Bash：

```bash
python -m pip install -r requirements-evolution.txt
export PYTHONPATH=src
python -m hmopt.evolution.cli --help
```

以下命令统一用 `python -m hmopt.evolution.cli`。完整安装本项目后也可使用 `hmopt evolve` 或 `hmopt-evolve`。`--root`、`--git-bin` 为顶层选项，必须写在子命令之前；所有流程命令使用同一个 root。Git 不在 PATH 时通过 `--git-bin` 指定可执行文件。

## 2. 先运行可检查的示例

```powershell
python -m hmopt.evolution.cli demo --output "$env:TEMP/hmopt-evolution-example-01"
```

输出目录必须不存在。示例仅创建并修改该目录里的 `example-repo`，演示历史提交中的临时 list 分配优化如何形成另一个文件的候选。它生成：

| 产物 | 用途 |
|---|---|
| `curated_pattern.json` | 带历史来源、适用条件和风险的示例规则 |
| `approved_plan.json` | 瓶颈分类、批准文件和冻结指标策略 |
| `plan_review_payload.json` | 方案及绑定其摘要的独立评审结构 |
| `implementer_handoff.json` | 批准实施后才可生成的任务交接包 |
| `code_review.json` | 绑定实际实施证据摘要的代码评审 |
| `synthetic_ab_report.json` | **手工构造的模拟数字**，不是性能测量 |
| `summary.json` | 状态、journal、审计事件数量和被阻止的操作 |
| `state/evolution.sqlite3` | 可重启读取的完整持久状态 |

示例作者、责任人和评审人都是测试身份。示例只执行简单 Python 功能断言，没有调用模型、构建内核或操作设备。预期最后仍为 `code_approved`，保留标记为 simulation 的验证结果；它不会进入真实 `validated`，也不能把模拟经验晋升 staging/hub。

## 3. 历史 → Pattern → 候选

对自己的目标仓库执行只读挖掘：

```powershell
python -m hmopt.evolution.cli mine "C:/work/kernel" --max-commits 200
python -m hmopt.evolution.cli list --kind pattern
```

每次读取从旧到新的 first-parent 提交页。重复运行同一目标仓库会从已导入游标继续，直到输出 `caught_up: true`。这能逐页遍历主干历史；不是遍历所有分支的每个提交。合并提交相对第一父节点取差异。force-push 后旧游标不再是祖先会失败，应由操作者判断后指定 `--no-incremental` 重新遍历。当前游标以本地绝对仓库路径为键，多分支独立采集使用独立 root。

评审记录目前通过 `import-history <records.json>` 导入，输入是 ChangeRecord 数组，`review_notes` 为导出的评审文本。来源标识必须稳定；同一来源 ID 的内容不可悄悄变更。查看具体契约：

```powershell
python -m hmopt.evolution.cli schema history
python -m hmopt.evolution.cli schema pattern
```

启发式蒸馏只产出草稿。策展者应阅读问题特征、诊断路径、修复方式、前提、风险和出处，将可用经验写为新 pattern 或新版本：

```powershell
python -m hmopt.evolution.cli check-contract pattern curated-pattern.json
python -m hmopt.evolution.cli import-pattern curated-pattern.json
python -m hmopt.evolution.cli activate-pattern "PATTERN_ID@1" --actor curator --note "已核对历史差异和适用前提，规则仅用于筛选符合这些条件的路径。"
```

`source_ids` 必须引用已导入 history 或 V2 source。草稿中的启发式/截断/未解决前提标记会使候选保留在 workbench；单纯激活不会让这些问题自动消失。真正完成语义策展后再创建完整的新版本，保留原有来源证据。

责任人文件 `owners.json` 示例：

```json
{
  "mm/**": "memory-owner",
  "kernel/**": "kernel-owner"
}
```

这里只是输入映射，不自动读取 CODEOWNERS。`*` 匹配单个路径段，`**` 匹配任意层级；不要假定根目录 `*.c` 能覆盖全部子目录。热点输入为对象数组，每项包含 `path`、`weight`（0..1）、完整 `revision`，可选 `symbol`。热点必须绑定本次扫描 revision，旧版本画像会拒绝使用。

```powershell
python -m hmopt.evolution.cli scan "C:/work/kernel" --owners owners.json --hotspots hotspots.json --top-k 20
python -m hmopt.evolution.cli show CANDIDATE_ID
```

漏斗先限定文件范围和字面特征，再结合来源、热点和责任人排序。`score` 是解释性排序值，不是正确概率；`pipeline`/`workbench` 是路由建议，均需经过后续门禁。缺少热点、来源前提未核实等候选转入交互工作台。没有 owner 的候选不能确认，需补充映射再扫描；尚未确认的快照可以刷新，已确认快照保持固定。

默认最多检查 5,000 个文件；可通过 `--max-files` 调整至最多 50,000。单次最多 256 个活跃 pattern、注册库最多 1,000 个版本，匹配结果最多 1,000 条，后续再做历史结论抑制和 Top-N。因此返回少量甚至零条不代表已扫完大仓。读取还受单文件/总字节/时长预算约束；V2 已增加 coverage 明细并标记部分覆盖，代码扫描仍没有持久化文件分片游标。规模化分片属于下一阶段。

## 4. 确认、方案、实施、评审

每个响应都有 `id`、`version`、`data`。提交时使用当前 version；发生冲突先重新读取。`request_id` 为一次逻辑操作的唯一标识，重试同一次操作时复用原输入和原 ID；改动输入需新 ID。

```powershell
python -m hmopt.evolution.cli decide CANDIDATE_ID confirm --actor OWNER --version 1 --request-id owner-confirm-001 --note "确认该候选值得开展方案设计，目标路径与责任范围已核实。"
python -m hmopt.evolution.cli handoff CANDIDATE_ID --output architect-handoff.json
```

确认后交接给 architect，此时 `source_changes_allowed` 为 false。确认、方案批准、实施前交接均检查目标仓库 HEAD 与候选基线一致且没有已跟踪文件的未提交修改；未跟踪文件不参与该检查。适配器应在隔离 worktree 工作，服务本身不提供 OS 沙箱。

方案契约可由 `schema plan` 获取。它必须指定候选、完整 baseline revision、作者、可验证假设、瓶颈分类、主指标选择理由、准确 `allowed_paths`（不支持通配符），以及冻结的设备/工作负载/环境和评价策略。

```powershell
python -m hmopt.evolution.cli check-contract plan plan.json
```

该命令返回规范化的 `contract` 和 `sha256`。评审者核对实际内容后，将此摘要填入 Review 的 `subject_digest`；不要直接计算原 JSON 文件字节的摘要，默认值和空白可能不同。方案审批的 payload 是 `{ "plan": <规范化方案>, "review": <评审> }`，参见示例生成文件。

```powershell
python -m hmopt.evolution.cli decide CANDIDATE_ID approve_plan --actor PLAN_REVIEWER --version 2 --request-id plan-review-001 --payload plan-review.json
python -m hmopt.evolution.cli handoff CANDIDATE_ID --output implementer-handoff.json
```

方案通过后，交接包才允许 implementer 在准确批准范围内改代码。外部 Agent/人完成实施并提交到目标仓库后，用包含 `{ "revision": "完整实施提交哈希" }` 的文件登记：

```powershell
python -m hmopt.evolution.cli decide CANDIDATE_ID record_implementation --actor IMPLEMENTER --version 3 --request-id implementation-001 --payload implementation.json
python -m hmopt.evolution.cli handoff CANDIDATE_ID --output reviewer-handoff.json
```

服务检查实施提交是批准基线的后代、确有变化、全部变化路径属于批准范围，然后保存真实差异和摘要。重命名按删除和新增路径校验。代码 Review 的 `subject_digest` 使用返回状态里的 `implementation_digest`，其 payload 就是 Review 对象，无需外套 `review` 字段。

```powershell
python -m hmopt.evolution.cli decide CANDIDATE_ID approve_code --actor CODE_REVIEWER --version 4 --request-id code-review-001 --payload code-review.json
python -m hmopt.evolution.cli handoff CANDIDATE_ID --output validator-handoff.json
```

| 门禁 | 当前约束 |
|---|---|
| 责任人确认/拒绝 | actor 必须等于候选 owner |
| 方案评审 | 评审者不同于方案作者，摘要与候选/基线匹配 |
| 实施 | 实施者不同于方案评审者，差异限制在批准路径 |
| 代码评审 | 评审者不同于实施者，绑定实施证据摘要 |
| 验证 | 验证者不同于实施者，报告匹配冻结策略与提交 |
| 知识策展 | 策展者不同于实施者和验证者 |

这些是可信本地操作者身份约定，CLI 的 actor 字符串不是认证凭据。MCP 不开放 owner/curator 操作，但能使用任意 shell 的 Agent 仍可能调用 CLI；生产强权限隔离需要独立服务身份、数据库 ACL 和受控执行器。

评审 decision=reject 会阻止状态前进，但当前不保存一次被拒绝的评审尝试。应通过 `capture expert_decision` 保存意见，责任人可用 `decide ... reject` 终止候选。修订代码需要新的候选/审批链，首期没有自动返工调度。

## 5. A/B 验证及再次测量

```powershell
python -m hmopt.evolution.cli schema ab-report
python -m hmopt.evolution.cli check-contract ab-report report.json
python -m hmopt.evolution.cli validate CANDIDATE_ID report.json --actor VALIDATOR --version 5 --request-id validation-001
```

baseline 和 candidate 必须分别对应批准基线及已评审实施提交，两侧镜像摘要不同，设备/工作负载/配置/环境一致且符合方案。指标集合、单位、方向、阈值、minimum_pairs 必须和批准策略相同。`functional_passed` 来自执行器汇总，本地服务不会自行编译或复跑功能测试。

主指标采用每对样本的方向归一化百分比，再等权计算均值与双侧 95% Student-t 区间。降低型指标的单对收益为 `100*(baseline-candidate)/abs(baseline)`，提高型方向反转。至少 3 对；主指标均值必须为正，区间下界达到批准阈值。护栏检查平均回退是否超出允许值。缺样本、设备不一致、无法量化或主指标噪声过大返回 inconclusive，明确回退或功能失败返回 fail。策略/提交绑定不符会直接拒绝报告，不能算一次有效验收。

这不是生产统计充分性的自动证明：样本需独立且有代表性，配对效应近似正态；热状态漂移、相关重复、工作负载偏差和多次试验挑选均需采集协议控制。大于 31 对时保守复用 df=30 的临界值。JSON 的 `hardware` 只声明来源，没有验签；`hardware_verified` 表示报告内部一致性检查，不能独立证明设备执行或镜像确实由该提交构建。

一次验证结果会封存，禁止直接覆盖。只有 inconclusive 可由 owner 用 `retry_validation` 加说明重新开放测量，旧结果保留在 `validation_history`，冻结策略和实施摘要继续有效。当前没有最大重试预算，生产接入应预先限定次数，不能反复抽样直到偶然通过。

```powershell
python -m hmopt.evolution.cli decide CANDIDATE_ID retry_validation --actor OWNER --version CURRENT_VERSION --request-id retry-001 --note "已核实测量中断原因，将按原设备和原指标策略重新采集完整配对样本。"
```

模拟只在 CLI 显式 `--simulation` 下运行；不会晋升真实 validated 或产生可晋升技能。不能修改示例 hardware 标记来替代真机结果。

## 6. 沉淀与回流

验证通过、失败、不确定和 owner 拒绝都会进入带证据的 journal。模拟记录不参与真实结论抑制。非模拟结论抑制同仓库、同文件源码摘要、同 pattern 版本的重复建议，即使无关提交导致 HEAD 变化也保持抑制。

```powershell
python -m hmopt.evolution.cli list --kind skill
python -m hmopt.evolution.cli recall "target path allocation" --limit 3
python -m hmopt.evolution.cli audit CANDIDATE_ID
python -m hmopt.evolution.cli evidence EVIDENCE_SHA256
```

六类知识信号为 `validation_result`、`failure_cause`、`effective_recipe`、`expert_decision`、`structural_fact`、`knowledge_correction`。补充知识必须引用已持久化 candidate、history 或 V2 source，始终先进入未验证 journal：

```powershell
python -m hmopt.evolution.cli capture failure_cause CANDIDATE_ID "画像显示瓶颈来自锁等待，指令缩减没有改善主指标，后续筛选应补充竞争证据。" --actor analyst
python -m hmopt.evolution.cli capture knowledge_correction CANDIDATE_ID "旧配方仅适用于指定配置，当前配置需要追加生命周期检查。" --actor expert --corrects SKILL_ID
```

纠正会保留关系和审计事件，当前不会自动覆盖旧知识或阻止旧条目召回。召回是跨全部持久记录的词面排序；中文未接入分词/语义检索，输出默认最多 3 项、硬上限 4 项，hub 在同分时优先。

```powershell
python -m hmopt.evolution.cli promote SKILL_ID staging --actor CURATOR --version 1 --request-id stage-001 --note "真实 A/B 达到冻结门槛，适用条件和副作用检查已完成，纳入团队试用。"
```

staging 需要真实通过记录及独立策展。hub 必须先 staging，并要求同一 pattern 版本至少两个已策展验证上下文、不同源码上下文和不同镜像对。这个规则是首期复现门槛，不意味着统计独立性、全面泛化或自动安装技能。知识内容只是检索证据，不作为 Agent 工具指令直接执行。

失效 pattern 可用 `retire-pattern PATTERN_ID@VERSION --actor ... --note ...` 停止该版及更早版本匹配。替代规则需新版本、历史出处和重新策展。现有已确认候选不会被退役操作自动取消，应由 owner 检查并拒绝不再适用的项。

## 7. MCP / 现有平台适配

本地 MCP 客户端启动命令：

```text
python -m hmopt.evolution.cli --root ABSOLUTE_STATE_DIRECTORY mcp-stdio
```

客户端需继承安装环境或设置绝对 `PYTHONPATH`，并传入可用 Git 路径。该入口使用 stdio，不启动 HTTP 端口。工具包括 list/show/handoff/submit/validate/recall/audit/evidence/capture，V2 新增 quality/convert_ic/dispatch。`submit` 仅允许 approve_plan、record_implementation、approve_code；没有 owner 确认/拒绝、重测批准、pattern 激活、知识晋升或模拟验证工具。

OpenCode 适配器应读取 handoff 的 role、state_version、allowed_paths 和 source_changes_allowed，映射到已有角色及工作目录，最后将不可变提交和证据交回本服务。构建/设备任务的“completed”不等于测试通过，必须检查嵌套退出码、产物和功能结果。真机适配器应由可信执行路径构造 ABReport，绑定构建 provenance、设备租约和实验协议。

首期交付的是可执行门禁及交接契约；尚未自动接通 OpenCode 调度、真实采集、分布式重试、身份认证和技能发布。接入顺序和验收标准见 [演进路线图](EVOLUTION_ROADMAP_CN.md)。

## 8. 本地验证

安装 pytest 后，以 Git 可用的环境运行独立测试：

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_evolution_mining.py tests/test_evolution_validation.py tests/test_evolution_service.py tests/test_evolution_learning.py tests/test_evolution_interfaces.py tests/test_evolution_entrypoints.py
```

这些测试使用临时仓库、临时数据库、构造的 A/B 报告和本地 stdio 子进程。真实设备与模型的验收应在生产接入后单列，不能用这里的通过数代替。

## 9. 本次实施验证记录

2026-09-10，在 Windows 本地临时 Python 环境完成：

```text
python -m pytest -q tests -k "not test_exec_command_not_found"
317 passed, 1 deselected in 108.04s
```

其中新 evolution 模块相关测试 197 项，原有测试 120 项。唯一未运行的旧测试是 `tests/test_windows_relay.py::test_exec_command_not_found`：它实际调用 `fastboot devices`，并假定 fastboot 未安装，不适合作为本次纯本地验证。其余设备操作测试使用 mock 或本地假 relay。新 MCP 测试实际启动 stdio 子进程，验证与 CLI 共用数据库、重启、角色门禁和错误响应，不连接生产服务。

新增/修改 Python 文件通过 Ruff 的语法和未定义名称等核心规则检查；新增模块和测试完成格式整理。学习测试在 fixture 导入整理后单独复跑，29 项通过。

独立执行 demo 验证了：临时仓库历史挖掘→策展规则→候选→模拟责任人/方案批准→示例实施提交→独立代码评审结构→功能断言→模拟 A/B journal。未确认交接和模拟技能晋升均被阻止。它证明本地协议可以执行，不构成一次内核优化收益、生产评审或真机验证。
