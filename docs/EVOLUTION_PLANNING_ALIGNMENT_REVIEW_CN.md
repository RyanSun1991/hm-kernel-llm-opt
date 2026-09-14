# Evolution 与原始规划的对齐审阅

后续实现说明：本文保留当时的审阅结论。历史代码分析与 Skill 流程、后台 worker 和审批桥已在后续增量中实现；
当前能力及仍待真实部署验收的边界见 [生产使用指南](EVOLUTION_PRODUCTION_RUNBOOK_CN.md)。

审阅日期：2026-09-10。起始代码基线：`opencode@bed7e77d4030ef2463ce2b930867c7e9710e42ca`。
审阅期间其他任务可能继续修改工作区；本报告中的代码定位以本次实际读取的实现为依据，交付前应结合后续 diff 复核。
方法：独立阅读用户原始要求、两份本地规划文本、当前设计与关键实现；本次只形成审阅记录，没有运行模型、设备或性能规模测试。

## 1. 结论

当前实现已经建立了**通用候选状态机、明确的责任人确认、可追溯证据门、原生工作台接入和严格的量化报告校验**。
统一 MCP、单文件配置、setup/doctor、七角色 recipe 与本地任务工作区，也符合用户希望复用平台能力、降低使用成本的方向。

但用户最初要求的能力还不能整体判定完成。**“自动从历史提炼可靠模式 → 结构/语义与热点筛选高价值目标 → 经实际 Agent 和真机完成优化 → 正负经验持续改进下一轮”仍有关键缺口。**
当前成熟度主要集中在流程控制与证据管理；候选供给、规模化遍历、运行时接入和词典自动演进仍以基础实现或人工操作为主。
现阶段适合称为“可进入受控试点的优化循环平台”，不足以称为已经具备经超大规模生产验证的自迭代优化能力。

## 2. 审阅依据与判断口径

用户的直接要求优先于附件内容：能力应围绕平台、框架与通用流程组织，内存/性能只是应用方向；循环需要覆盖挖掘、筛选、确认、实施、验证、沉淀。
后续还明确要求统一 `src/hmopt/api/` MCP、OpenCode 交互触发、专家确认机制以及简化配置。

本地附件实际可读取的范围为：

| 来源 | 本次读到的内容 | 限制 |
|---|---|---|
| 用户会话中的规划原文 | 通用 Multi-Agent Harness 演进、历史提交与评审、pattern/反模式、漏斗/热点、责任人、真机 A/B、Skill Hub | 这是目标与参考方向，不是当前代码能力证明 |
| `C:/Users/irtos/.codex/attachments/7df2ca6a-7df4-43a2-a73f-1dfea10a0efa/pasted-text.txt` | 100 行；v0.9 标题、目标/非目标、阶段表、§1、§2 架构图前半 | 文件止于架构图，图和后续章节不完整 |
| `C:/Users/irtos/Downloads/plan.md.txt` | 144 行；从“pattern 统计重算”列表开始，包含 §6–10 与附录 A–C | 虽被称为完整版本，当前磁盘文件没有前置章节；与前一附件合看仍缺 §2 后半至 §5 的完整正文 |
| 当前代码和仓库文档 | `src/hmopt/evolution/`、`src/hmopt/api/`、工作台 recipe、对应测试和操作指南 | 文档中的拟议接口不算实现；fixture 和接口测试不算真实 Agent/真机执行 |

本次未读取 PPT，也不推断缺失章节。关于外部研究或 Google ECO 的生产规模，本审阅没有独立核验；它们不能作为本项目已经达到同等规模的证据。

状态定义：**已实现**表示对应程序能力存在；**部分实现**表示有可用基础但目标链路未闭合；**缺失**表示未找到对应实现；**未验证**表示接口/契约存在，但没有本次可核查的目标环境执行证据。
P1 表示影响原始核心目标或正式生产验收，P2 表示规模扩展、共享或运维阶段需要补齐；优先级不代表发现了安全事件。

## 3. 对齐与差距矩阵

| 规划能力 | 当前判断 | 可核查实现与限制 | 优先级 |
|---|---|---|---|
| 通用平台、流程与多业务验收 | 已实现核心 | [Plan/ValidationPolicy](../src/hmopt/evolution/service.py#L40) 支持瓶颈分类、任意指标、性能与 correctness；通用角色不按内存/性能新增两套执行器。源码后缀、历史关键词等仍在挖掘器内硬编码，扩展领域尚需适配 | P2 |
| 主 MCP 统一入口及单步骤暴露 | 已实现 | [主注册器](../src/hmopt/api/mcp_service.py#L831) 组合 5 个索引工具与 [19 个 Evolution 工具](../src/hmopt/api/evolution_mcp_service.py#L50)，stdio/HTTP 共用；旧入口保留兼容 | — |
| 一份配置、可执行的使用入口 | 已实现 | [setup](../src/hmopt/evolution/setup.py#L136)、[doctor](../src/hmopt/evolution/doctor.py#L137) 和 [配置加载](../src/hmopt/api/evolution_mcp_service.py#L26) 已落地；生成 pilot 和连接片段，不改 provider。doctor 只做预检查 | — |
| 规模化、增量历史遍历 | 部分实现 | [mine_git_history](../src/hmopt/evolution/mining.py#L307) 有第一父链分页、冻结 revision、限额、截断标识；逐提交串行执行多次 Git。未实现原规划的并行挖掘，未证明 5 万提交/10 分钟目标 | P1 |
| 历史评审、实验与团队知识来源 | 部分实现 | [ChangeRecord](../src/hmopt/evolution/mining.py#L83) 可接收 review_notes；[本地来源](../src/hmopt/evolution/sources.py#L121) 与 [原生记忆导入](../src/hmopt/evolution/native_memory.py#L201) 可分页。Git 导入不会自动拉取托管平台评审线程，未找到 Gerrit/PR 评审采集适配器 | P1 |
| 蒸馏问题—诊断—修复 pattern 与反模式 | 部分实现 | [Pattern](../src/hmopt/evolution/mining.py#L133) 有结构化三元组与适用条件；[Git 蒸馏](../src/hmopt/evolution/mining.py#L470) 取关键词和最多三条删除行，[来源蒸馏](../src/hmopt/evolution/sources.py#L621) 取文档代码片段。缺跨记录语义聚类、问题/修复角色判别、revert 关联与可选 LLM 蒸馏通道 | P1 |
| 全库范围→锚点→结构谓词→语义 judge 漏斗 | 部分实现，结构/语义阶段缺失 | [Matcher](../src/hmopt/evolution/mining.py#L108) 只有 glob 与 literal all/any/none；[扫描器](../src/hmopt/evolution/mining.py#L598) 在整个文件文本上判断。没有函数/循环结构定位、数据流/调用链谓词、SCIP 特征 provider 或 judge 阶段 | P1 |
| 热点对齐、memory-bound 分类、趋势与回归定位 | 部分实现 | [Hotspot](../src/hmopt/evolution/mining.py#L156) 接收调用方准备的 path/symbol/weight/revision，拒绝旧 revision；不是常态化 profiler 摄取、自动瓶颈分类或版本 bisect。当前 Plan 中的 bottleneck 仍由方案作者提供 | P1 |
| 高置信候选、收益排序与证据简报 | 部分实现 | 候选保存源码摘要、行、来源、score breakdown；但 score 是字面命中/来源数/热点/owner 的固定加权，不是正确概率、独立确认的语义适用性或收益估计。同文件同 pattern 最多产出一个候选 | P1 |
| 责任人规则、确认单、禁止越门 | 已实现本地合同 | [owner 确认](../src/hmopt/evolution/service.py#L448) 校验已绑定 owner、候选版本与基线；[确认单](../src/hmopt/evolution/workflow.py#L113) 保留上下文并逐项应用。owner/actor 是本地标签，没有认证身份 | — |
| 专家地址簿、消息/邮件、审批回执、续跑 | 缺失 | 只有路径→owner 规则。当前 [审批桥文档](EVOLUTION_APPROVAL_BRIDGE_DESIGN_CN.md) 明确相关通道和回调均未实现。最早草案不要求 headless，但后续用户提出的通知/回复闭环仍未满足 | P1（若要求无人值守） |
| Agent 研究→方案→实现→独立评审 | 接入已实现，实际运行未验证 | [任务步骤](../src/hmopt/evolution/workflow.py#L312) 和 [execution skill](../.opencode/skills/infra/pipeline/evolution-execution/SKILL.md) 定义真实 OpenCode task 委派与阶段续跑；Python 只派发工件。没有可核查的真实模型完成日志可用来宣称已经跑通生产执行 | P1 验收 |
| 程序化 A/B 与代码/方案双评审门 | 已实现报告合同 | [transition](../src/hmopt/evolution/service.py#L448) 绑定方案/实现摘要和独立作者；[lmbench 转换](../src/hmopt/evolution/lmbench.py#L237) 读取 raw xlsx、校验 profile/hash/配对；[evaluate_ab](../src/hmopt/evolution/validation.py#L148) 判断功能、配对、置信区间与 guardrail | — |
| 真机来源证明与真实优化收益 | 未验证；部分证明链待补 | A/B 身份和 `hardware`、`functional_passed` 来自报告声明；可核查 raw 字节不等于证明谁在何设备采集。当前没有实际试点结果证明目标内核收益、设备调度与重试闭环 | P1 |
| 正向经验回原生 Team Memory/Hub | 部分实现 | [export_native_memory](../src/hmopt/evolution/native_memory.py#L376) 校验报告为非模拟通过及独立 curator，输出一个原生 fact/validated journal 与 L1 staging，遵守原生 schema/脱敏；未自动产出 landed idea 或完成共享发布 | P2 |
| 失败、专家否决、反例形成共享反模式 | 部分实现，原生负向出口缺失 | [_journal](../src/hmopt/evolution/service.py#L744) 保存失败/拒绝，capture 可收六类信号；但 [_proposal](../src/hmopt/evolution/learning.py#L324) 只允许 curated pass，当前 native 导出固定 fact/validated，无法导出 bad_plan/anti_pattern/idea rejected | P1 |
| 质量统计驱动词典自调优与晋升 | 部分实现 | [quality](../src/hmopt/evolution/learning.py#L172) 重算真实事实并区分模拟；[overlay](../src/hmopt/evolution/learning.py#L257) 由操作者手动设置 factor/probation。无自动阈值策略、自动恢复/退役、pattern PR 或 technique 晋升信号生成链 | P1 |
| 周期迭代、跨成员稳定身份和去重 | 部分实现 | discovery 批次可恢复，结果指纹可抑制本地重复；没有 Evolution nightly/中央候选调度。repo/candidate 身份含 checkout 路径，[结果指纹](../src/hmopt/evolution/service.py#L129) 也含 repo_path，因此不满足不同成员相同目标得到同一 ID 的规划 | P2 |

## 4. 影响核心目标的具体差距

### 4.1 现在能够生成线索，但还不能自动蒸馏高质量优化方法

Git 蒸馏器的主要规则是：提交文字含性能/修复关键词，选择删除行作为 required literal，生成 draft。
对应 `problem/diagnosis/remedy` 主要是通用说明，不是从历史材料中恢复出的根因、修复过程与已验证收益。
技术文档蒸馏同样无法判断一个代码片段是错误示例、正确修复还是上下文；实现对此诚实标注了 HEURISTIC/UNRESOLVED。

因此，自动挖掘目前减少的是“找哪些文字线索”的工作，尚未大幅减少“证明为何适用和怎样修复”的专家工作。
默认 instruction_count 草案也不是 memory-bound 路径的有效指标选择，应继续要求策展和方案阶段重选评价依据。

优先补齐结构化变更/评审输入、修复前后片段绑定、revert/拒绝原因分类、支持样本聚合和反例，再提供有预算的可选模型蒸馏。
验收应包括多条真实提交归并为同一机制、正反例分离、拒绝把已修复代码重新认作错误，以及新蒸馏 pattern 在独立目标上产生有效候选。

### 4.2 当前漏斗是文件级字面筛查，语义与热点交叉尚未程序化

扫描器检查 `literal in content`，再用首个 required literal 的位置截取最多 4096 字符；同 pattern 的多个目标不会逐一枚举。
例如，同一文件中的两个函数分别包含两个 required literal，也可能通过文件级 all_of；一个函数里的 none_of 则可能抑制另一个函数中的有效目标。
这些是当前粒度的直接结果，不意味着程序已经提供函数/循环级语义判断。

热点的 symbol 只通过“是否出现在 excerpt 中”辅助匹配，没有版本固定的符号定位与调用关系证明。
现有索引 MCP 已存在，但目前主要由 researcher 在责任人确认之后按 skill 使用，尚未成为候选确认前的自动漏斗阶段。
因此，人工确认仍承担大量筛查职责，难以直接获得用户期望的高置信 Top-N。

应先实现可定位的 target/anchor 模型及多个匹配实例，再接版本绑定的 SCIP/结构特征和热点证据。
语义复核可以逐级降级，但必须记录该候选走过哪些阶段、哪些条件没有证明；不要将未执行的阶段折算成信心分。
收益排序应单独表达预估收益、成本和不确定性，保留当前 score 为检索启发式，不能把二者合并成未经校准的概率。

### 4.3 “需要分区”已有状态，但缺少可直接执行的分区扫描能力

历史分页有真实 cursor；源码扫描不同：[扫描实现](../src/hmopt/evolution/mining.py#L670) 先列 Git tree，再处理 `rows[:max_files]`。
配置上限为 50000 个 tree entries、扫描输入 64 MiB，逐个 blob 启动 Git；[DiscoveryConfig](../src/hmopt/evolution/discovery.py#L19) 没有文件 cursor、partition manifest 或枚举 pathspec。
超过限额后 [discovery](../src/hmopt/evolution/discovery.py#L289) 提示 `requires_partition`，但同一根仓库重复调用仍从同一位置开始。
只缩小 pattern 的 file_globs 也不会改变 tree entries 的前缀截取方式。

因此，“明确报告没扫完”已经做到，“在任意大型仓库上按当前接口分区续完”尚未做到。
另外，历史导入每个 draft 都读取整个 pattern registry 查找版本；随着历史和版本增加，需要单独评估这一存储路径。

建议先增加冻结 manifest、路径/对象游标、跨分区覆盖汇总，再使用 Git batch I/O 和有界 worker 提高吞吐。
验收使用超过现有 tree/字节限额的仓库验证无遗漏、无重复、断点恢复和完整 coverage；随后才测试原规划的 5 万提交/10 分钟、2 万 C 文件×12 pattern/5 分钟目标。
本次没有实测这些耗时，不给出达标承诺。

### 4.4 反馈已有审计数据，但尚未成为自迭代的完整输入

当前失败与拒绝留在本地 Evolution 记录中；主动 capture 的经验是 unverified，不能绕过晋升门。
这是正确的证据约束，但 native 导出的实现只接受成功的已策展条目，负向经验没有独立的共享通道。
原生知识导入保存原文，却把 `decision_reason` 设为 unknown；其状态和拒绝分类尚未直接驱动差异化反模式蒸馏。

应新增与“成功方法可推广”分开的“失败观察/坏方案/反例”导出合同：保存失败层次、适用范围和不确定性，经 curator 审核进入原生 anti_pattern、bad_plan、idea rejected 等记录类型。
失败不能被写成 validated 修复，但也不应因为不成功而永远无法沉淀。
优先验证：一条被专家拒绝或 A/B 回归的候选，可以在下一目标的筛查中触发解释清楚、范围正确的降权/抑制。

`quality` 返回的统计很有价值，但 overlay 是人工参数，不会随下一轮统计自动变化。
可以先生成可审查的调优建议和应用记录，再逐步对低风险降权开启自动策略；恢复、退役和新版本激活应有各自条件。
自动策略必须基于独立标签、样本量和失败原因，不能原样沿用附件中的 `confirmed/(confirmed+rejected)` 并称为 precision。

### 4.5 Skill Hub 已接到正向出口，pattern 本身还不是完整的共享资产

当前原生出口产生一个 fact 候选，后续策展/CI/release 继续使用已有 Hub；内部 tier=hub 也明确不代表正式发布。
但早期规划要求的可执行 pattern 目录、schema/lint、pattern 晋升 PR、≥2 目标验证后的 technique 信号，尚未形成同等完整链路。
内部 `_proposal` 输出的摘要清单不能替代可安装方法或共享 pattern 资产。
现有 [promote](../src/hmopt/evolution/service.py#L794) 检查多个 source_context 和 image_pair，是内部晋升约束；
source_context 包含文件内容摘要，同一逻辑目标不同版本也可形成不同上下文，不能直接当作“两个不同目标独立复现”的证明。

建议分别管理可执行 pattern 与叙事知识：pattern 有版本、支持/反例、匹配器测试和性能预算；知识仍走已有原生 schema。
以一次完整的 pattern 提案→CI/评测→发布→另一成员导入→新目标命中作为验收，而不是只检查本地 staging 文件存在。

### 4.6 “Agent 可以被触发”和“真实优化已经验证”仍是两个层次

统一 MCP、官方协议接入测试、工作台 role/skill 合同与服务端证据门已实现；这解决了平台接入问题。
真实 OpenCode runtime 是否按该 recipe 完成多角色交接、代码提交、构建、烧写和采集，需要会话及执行工件验证。
当前 [操作指南](EVOLUTION_OPENCODE_RUNBOOK_CN.md#10-本次验证记录2026-09-10) 记录的是离线测试与本仓库 133 条第一父链历史发现；零 active pattern 时零候选不能证明候选有效性。

量化门对冻结 lmbench profile 的 raw 来源校验比附件中的 digest/validation.md 方案更强，应保留。
同时，[hardware_verified](../src/hmopt/evolution/validation.py#L302) 基于报告中的 hardware 声明，functional_passed 也是输入布尔值；这些字段不提供设备或执行器认证。
通用 ABReport 在未冻结 measurement_profile 时仍允许其他来源，不应把“所有 performance 验证都强制来自已采集 raw xlsx”作为平台保证。

正式试点应让现有测试采集服务生成带 run/build/image/source 关联的机器可核查 manifest，保留功能检查输出与设备执行日志，避免由 Agent 自报这些关键事实。
按原规划至少完成 20 条真实专家裁决和 3 个完整候选，再判断流程质量和验证成本；没有收益的结果也要如实统计。

### 4.7 自动通知与持续调度应单独验收，不能由 owner 或 full 参数隐含获得

最早草案明确不做 headless OpenCode，所以没有自动启动执行引擎本身不是偏离其 M1 非目标。
但后续用户询问“发送给谁、怎样发送、怎样处理回复并继续”，使审批桥成为需要明确交付边界的产品需求。
目前 owners 只确定责任标签，不提供账号、地址、身份核验或回调；确认后仍要显式触发候选入口。

自动模式需要目录版本、可靠投递、认证决定、过期版本拒绝、幂等回执与原工作台会话/scope 绑定。
周期 discovery 还需要批次租约、失败恢复、无变化处理和预算记录；原生 Hub 的 nightly 不应被误认为 Evolution 已有调度。
保留现有手动入口作为可用模式，同时明确自动模式何时完成，不用“配置好邮箱即可”掩盖尚无实现的适配层。

## 5. 应保留的设计调整

以下差异有助于用户目标，不建议为了表面对齐草案而回退：

| 当前选择 | 对齐理由 |
|---|---|
| SQLite/WAL、CAS、请求幂等与内容寻址证据，替代一候选一 JSON 文件 | 更适合可恢复、可审核的候选状态控制；是否分布式是后续扩展问题 |
| 使用既有七角色及 task-local workspace，不复用 current_task 单例 | 保留工作台默认行为，降低多候选互相污染的风险 |
| owner 必须显式配置，不自动把 git blame 邮箱当审批人 | 最后修改者并不等于有审批权限的专家；后续应接真实责任目录 |
| draft 与 active 严格分开，种子只是研究模板 | 不能把没有来源和反例的示例词典称为生产规则 |
| 冻结主指标/环境/profile，读取 raw 并校验配对 | 比自动选择“上一份”结果或解析 PASS 文本更可靠 |
| 性能与 correctness 两类验收 | 使通用诊断/可靠性能力不被强行套入内存和性能二分 |
| acceptance_rate 与 true_precision 分开；模拟/无效报告单列 | 避免用专家意愿或 fixture 成功率冒充真实识别质量 |
| 正式 Hub 发布仍经原生治理 | 自迭代不等于无人审核地修改团队共享方法 |

## 6. 建议的落地顺序与验收

先保留现有控制面，补齐候选供给和真实闭环，再扩大自动化范围；不再新增一套平行 Agent 框架或状态服务。

| 顺序 | 工作包 | 可检查的退出条件 |
|---|---|---|
| 1 / P1 | 固定一个试点场景，完成真实工作台/真机贯通 | 保存真实会话、各角色工件、评审摘要、提交与 raw 采集关联；按原 M2 目标形成至少 20 条专家裁决、3 条完整执行记录，收益据实记录 |
| 2 / P1 | 来源适配与结构化蒸馏 | 打通实际评审数据；支持问题/修复/反例区分；一个新蒸馏 pattern 经策展后在不同目标上产生已验证候选 |
| 3 / P1 | 结构/语义与热点漏斗 | 函数/循环目标独立枚举；冻结索引与运行数据版本；正负标签集能分别报告各阶段覆盖、误报、漏报和成本 |
| 4 / P1 | 可续完的大仓库扫描 | 分区 manifest 与游标在超过当前文件/字节限制时仍可完成；恢复不重复、不漏扫；再报告原规划规模目标的耗时与资源 |
| 5 / P1 | 负反馈、质量策略和 pattern 共享 | 失败/拒绝形成正确类型的原生候选；降权/抑制可追溯；至少一次 pattern 发布被另一任务消费，并验证下一轮变化 |
| 6 / P1 或 P2 | 通知审批与周期运行 | 若部署目标是无人值守，此项为 P1；验证真实专家身份、重复/过期回执、同 scope 续跑、重启恢复及连续 3 轮周期记录，否则保留显式手动模式 |
| 7 / P2 | 跨成员稳定身份与中央治理 | 相同逻辑仓库/目标在不同 checkout 得到同一逻辑 ID；运行实例仍独立；中央去重与认领不发生重复实施 |

具体技术验收不要用“测试总数达到 N”替代上述业务证据。规模测试、专家标签集、失败反例和真机收益必须分别有可复查工件。

## 7. 本次审阅的验证边界

本次检查了代码中的实现路径及相应测试定义，包括历史分页、字面筛查、owner/CAS、独立评审、lmbench profile、原生 staging、统一 HTTP/stdio 与 setup/doctor。
例如 [扫描测试](../tests/test_evolution_mining.py#L189)、[确认单测试](../tests/test_evolution_workflow.py#L54)、
[profile 绕过拒绝测试](../tests/test_evolution_lmbench.py#L414)、[质量统计测试](../tests/test_evolution_quality.py#L60)、
[原生导出测试](../tests/test_evolution_native_memory.py#L258)、[官方 HTTP MCP 测试](../tests/test_evolution_unified_http.py#L233)。

这些用例证明仓库有对应的合同验证覆盖；本次没有重新执行全量回归，也没有把已有单元/协议测试结果转述成真实候选精度、Agent 成功率或生产收益。
报告中的“缺失”限定于本次审阅的代码路径与可见配置，不推断外部团队已经拥有但尚未接入的系统。
