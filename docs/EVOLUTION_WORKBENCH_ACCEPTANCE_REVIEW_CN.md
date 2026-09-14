# Evolution 工作台实际接入与验收审阅

后续实现说明：本文是下述时间点的运行证据。新增后台 worker、责任目录、通知和认证回调的当前用法见
[生产使用指南](EVOLUTION_PRODUCTION_RUNBOOK_CN.md)；其本地协议测试不等于补做了本文中的真实模型或真机验收。

状态：**本地受监督 correctness 流程及原生 L1 暂存已完成；正式 provider、最终提示完整重放、真机与规模化能力仍待验收。**
审阅日期：2026-09-10 至 2026-09-11。最终运行时快照：`2026-09-11T06:39:57-08:00`，候选为 `validated` v6，原生导出为 staged / not_published，主会话已正常返回。
本报告记录真实 OpenCode runtime、统一 MCP、本地测试仓库与受监督模型桥的实际行为，并对照[规划对齐审阅](EVOLUTION_PLANNING_ALIGNMENT_REVIEW_CN.md)。

## 1. 本次能确认到哪一层

**实际工作台接入已得到比单元测试更强的验证：OpenCode 成功调用统一 MCP、执行有界发现、展示草案与候选，并在未确认候选上阻止 full 进入实施。**
运行时验证还发现了角色权限规则顺序问题；修复后，OpenCode 的 write 工具实际写入了允许的测试工件目录。

责任人随后通过 operator CLI 明确确认了一个隔离 correctness 候选。2026-09-11 的补充取证已确认：
researcher 与 architect 完成真实工件写入，方案评审已通过服务门；implementer 实际修改并提交唯一批准路径 `counter.py`。
`recorded-implementation.json` 已记录新提交 `d813a4c00076875b5fc7e2ed6228bf8f822c9127`，候选进入 `implemented` v4。
随后代码评审通过服务门，validator 在该提交上实际执行冻结检查；服务接受 `verdict=pass`，候选进入 `validated` v6。
独立操作者标签 audit-curator 再通过真实 CLI 策展自动捕获的结果，写入 1 份原生个人 journal 与 1 份 L1 staging；没有发布共享 Hub。
这证明本轮隔离正确性案例完成了发现、确认、方案/代码评审、实施、验证与本地沉淀；仍不代表无人值守或生产优化验收。

本次没有验证无人值守 provider 接入、真实内核规模化优化、设备烧写、真机 A/B 或生产收益。
本地模型桥由当前 Codex 监督者逐轮提供回复，真实工具执行由 OpenCode/MCP 完成；这是受监督的 runtime 集成验收，不是独立生产模型的自主执行能力评测。

## 2. 版本、环境与证据范围

| 对象 | 实际值与判断 |
|---|---|
| 首次推送检查点 | `716bfdc30ee5d962e0382de91e5bb7a72d0d6137`；按请求先推送此版本，再继续实际审阅 |
| 最终交付内容 | 本报告所在功能提交整合七角色、六个 profile 的权限修复与收窄、权限语义测试、四份通用技能与角色策略修正；最终提交及远程核对值记录在交付回执 `final-git-push.json`，不自引用提交哈希 |
| OpenCode | 本地真实 `opencode.exe`，会话元数据版本 `1.18.30`；所服务前端 bundle 的版本常量为 `1.18.29`，二者分别记录 |
| OpenCode 服务 | `http://127.0.0.1:49174` |
| 主会话 | `ses_f72e2b23bffemLnAH461iNuoyp`，标题“Evolution 工作台完整流程验收” |
| 模型连接 | `codex-audit`，指向本地受监督兼容桥 `127.0.0.1:49173`；没有访问真实外部 provider |
| 工作台与目标 | 两个独立临时目录：`AUDIT/workbench` 和 `AUDIT/target`；目标不是生产内核仓库 |
| 验收策略 | 本地 correctness；冻结六个检查，只允许修改 `counter.py`，没有性能提升百分比 |

`AUDIT` 在本机指向：

```text
C:/Users/irtos/AppData/Local/Temp/hmopt-opencode-audit-20260910
```

报告编写者通过只读文件、OpenCode 会话 API 和状态快照取证。**被审阅的整个验收过程包含真实写操作**：
OpenCode 执行 read/write/bash/MCP 等工具，另有单独记录的操作者策展、owner 确认与有界执行授权。
本地模型桥提供模型侧回复和工具调用请求，不返回预制工具执行结果；工具回执来自实际 OpenCode、Git 和 MCP 服务。
隔离测试仓库是验收对象，不能与“用工具结果夹具替代真实运行”混为一谈；同样，真实工具运行也不消除人工监督和操作者身份边界。
临时目录本身不是随仓库发布的证据包；必要的脱敏回执与工件另做本地白名单归档，见第 9 节。

## 3. 已观察的实际行为

| 检查 | 实际结果 | 证据与边界 |
|---|---|---|
| 不提供 discovery profile | assistant 查询已注册 pilot 后停止，要求明确输入 profile | `command-missing-profile.json` 与实际 MCP 工具记录；没有替用户猜一个 profile 后执行 |
| 首次 `/evolve-discover pilot` | 批次 `discovery-ab114b4d0a0d49d9af1685afd7d2de9a` 到 awaiting_review；4 条 Git 提交、1 个 draft、6 份工作台来源 | `command-discover.json`、`session-messages.json`；8 份模板/README 被跳过，来源未产生可用字面草案；active pattern 为 0，代码筛查未实际执行，候选 0 |
| 策展后新批次 | 显式导入并激活一个经本地规格和历史修复检查的诊断 pattern，再运行新批次 | `operator-curation.json` 三步命令均 exit_code=0；是操作者策展，不是 Agent 自动证明/激活新方法 |
| 已激活字面 pattern 命中 | 批次 `discovery-10657baae35b4023a45adbaaf5b9cf5c` 产出 1 个 counter.py:6 候选，score=0.55、lane=workbench | `discovery-after-curation.json`；没有热点输入，只扫描激活规则范围内 1 个文件，分数不是正确概率 |
| discovered 状态请求 full | coordinator 读取状态后停止，没有 dispatch、角色实施或源码修改 | `candidate-before-confirm.json`；等待指定责任人、实际版本和理由 |
| 责任人确认 | operator CLI confirm 成功，候选从 v1/discovered 变为 v2/confirmed | `operator-confirmation.json`；标签 audit-owner 明确限定隔离本地试验，不授予生产或设备权限 |
| 确认后请求 full | 研究、方案、双评审、实施与真实 correctness 验证已完成 | 当前候选 `validated` v6；完整执行期间存在明确记录的人工传输恢复和逐次权限授权 |
| researcher 实际完成 | 核对源码、规格、冻结基线和当前 HEAD，通过 write 写出 research-note.md，最终消息 finish=stop | 会话 `ses_f72cc1140ffeN0TVtp2Tr5Fr4K`；范围为局部 correctness，未改源码、测试、策略或服务门 |
| architect 实际完成 | 读取合同和研究证据，通过 write 写出 plan.json、plan.md、decisions.md，最终消息 finish=stop | 会话 `ses_f6f57aa49ffe4Le6g07VNjtnhx`；方案比较 strip 判定与显式 isspace，保持非法文本异常；初产物为草案，随后经独立方案评审获批 |
| 方案服务门 | plan reviewer 对规范摘要 `501d158e9a0033cc96c21ff0551fb3f6a57bae2672c0091258860964e0fad8e7` 评审，approve_plan 接受后到 v3/plan_approved | 初始 dispatch 的 execution submissions 与 `recorded-implementation.json` 中 plan_review；批准只开放实施门 |
| 真实 Git 实施 | implementer 修改并提交 counter.py，record_implementation 被服务接受，候选到 v4/implemented | `recorded-implementation.json`、implementer 会话 `ses_f6f4813deffeUi8UCjo4vdc1vC`、实际 Git commit/diff；实施记录与后续验证门分别留证 |
| 代码评审与验证 | 新 reviewer 批准精确实施摘要；validator 实际执行新提交，六项全部通过，evolution_validate 被服务接受为 v6/validated | validator 会话 `ses_f6f36952effeM4FR9jkIFRUhaY`、原始 candidate stdout、capture-summary 与 correctness-report；baseline 的失败来自准备阶段真实执行并在验证阶段核验 |
| 原生本地沉淀 | 自动 validation_result journal 经 audit-curator CLI 提升到 staging，导出为个人 journal 与原生 memory_item L1 staging | `native-curation.json`；schema 校验通过，3 个来源解析成功，同 request 重放未重复文件，未生成 knowledge/skills 发布内容 |
| Git 身份失败与重试 | 首次 commit 返回 Author identity unknown；第二次以单条命令的测试身份成功提交 | 实际 bash 回执；采用 `git -c user.name=... -c user.email=... commit`，未执行 Git 提示中的 global config；此身份是测试标签 |
| 主机恢复后续接 | 最新重试请求得到监督者回复后，原 researcher 完成，继续进入方案与评审 | 使用既有 session/dispatch；未创建第二个 dispatch，未重复 owner 确认；是人工模型传输恢复，不是无人值守恢复证明 |
| 原角色权限失败 | coordinator 的 write/edit/apply_patch 没有暴露，测试工件不能按预期写入 | `permission-session-messages.json`；没有通过 bash/MCP 绕过角色限制 |
| 权限顺序修复后 | 新会话的 write 工具返回 completed / Wrote file successfully，probe.txt 内容为 `permission probe` | 会话 `ses_f72d429b7ffebSLmgddpSIbU8G` 的实际工具记录，加本地文件交叉核对；各阶段角色另有真实执行回执，6 个 profile 组合没有逐一完整运行 |
| 最终权限收窄后 | 重新加载最终资产，13 个定义与真实 GET /agent 一致；工件 write 和固定 status 完成，带 --output 的 diff 要求审批 | 探针会话 `ses_f6f19339dffey5G0nfCo5fmciX`；监督者按测试安排拒绝该调用，输出文件未创建；没有执行破坏性命令 |
| 真实页面检查 | 根任务通过 CUA 检查实际 tab 4 的可访问性树，看到完整用户命令、长模板内容及当前 Thinking 状态 | 证据已超出仅收到 SPA HTTP 200；没有把页面可读与状态可见等同于所有 UI 交互通过 |

候选 ID：

```text
candidate_954afe880abeba5a3a46648e2274cad6d884a2473c78f94ee68c445c836c695d
```

命中依据是已提交源码中的 `return int(raw) if raw else 0`；测试规格要求空白字符串也归零。
`baseline.stdout.json` 中 `whitespace_to_zero` 实际失败，其余五项通过；`historical-fix.stdout.json` 中六项均通过。
这些文件证明隔离夹具中存在可复现问题和历史修复参考，**历史修复通过不能代替本轮新提交的独立验证**。

## 4. 权限规则修复与剩余隔离边界

### 4.1 运行时失败的原因与修复

此前部分角色把广义 `*` 规则放在具体路径规则之后。OpenCode 的实际规则语义是最后匹配优先：
末尾 `edit: {"*": "deny"}` 既覆盖目标工件路径 allow，还可导致 write/edit 类工具直接从模型可见工具集合中消失。
同样，bash 末尾的通配规则会覆盖前面的具体 allow/deny；implementer 的末尾通配 allow 也影响原本的特定禁止项。

首轮修改把通配兜底放到前面、具体规则放到后面，恢复预期的具体路径判定。
[权限回归测试](../tests/test_workbench_permission_precedence.py) 按所核对的 OpenCode 语义检查实际匹配结果与工具可见性，覆盖角色、profile、只读命令、源码 edit 和特定禁止命令。
本次没有通过真实执行破坏性命令来测试这些禁止项。

真实 runtime 的 probe 写入已证明 coordinator 的工件写路径恢复。该修复与后续权限收窄一并纳入本报告所在的功能提交；早期检查点不包含这些修复。

后续副作用用例进一步发现，旧的 `git diff*` 等前缀 allow 会放行 `git diff --output=counter.py` 这样的写文件命令。
产品随后将 12 个非 implementer 角色/profile 收窄为有限、完整的固定只读命令：额外参数、rg/find 等回到 ask，优先使用 read/grep/glob 工具。
implementer 原有默认 bash allow 与四项 deny 保留，这仍不是 shell 或操作系统 sandbox。
相关测试 **81 passed，6.06 秒**，Ruff、diff 检查及 14 个文件的 LF 检查通过。
完整候选会话结束后，将最终 17 份角色/技能资产复制到隔离运行副本并重新加载，真实 GET /agent 核对了 13 个角色/profile 定义。
新会话实际 write 成功、`git status --short` 无需审批完成；`git diff --output=...` 触发审批并按测试安排被监督者拒绝，输出文件不存在。
`final-runtime-permission-check.json` 与完整会话回执记录这些结果；这验证具体运行路径，不能推导出所有命令形式均受沙箱隔离。

### 4.2 共享工作区不是严格角色隔离

本次读取的 architect/reviewer/coordinator 等 frontmatter 允许 `.opencode/local/**`，覆盖所有 workspace 及其工件子目录；
测试也验证同一角色可以写共享 workspace 路径，并没有按 `artifacts/research`、`artifacts/review` 等实施角色独占访问控制；上层角色文字现已移除严格独占 ACL 的表述。
因此 reviewer 的角色说明虽然要求不改被评审工件，runtime 路径权限本身并不能阻止它修改另一个角色的工作区文件。

服务端对 owner、方案作者、评审者和实施者的 actor 差异，以及 plan/implementation digest 的检查，属于状态与证据合同。
这些检查不提供经过认证的人员/执行器身份，也不是文件系统权限、进程隔离或不可变工件存储的替代物。
尤其本轮各角色模型回复由同一监督链路提供，不能据此声称验证了真正独立的认知评审。

若部署需要独立执行隔离，下一步应把角色/任务限定的目录权限、不可变提交工件、执行器身份和审计关联作为单独验收项。
在此之前，准确的定位是“可信操作者环境下的角色协作合同和服务证据门”。

### 4.3 通用角色技能中的场景假设修正

实际 implementer 读取暴露出通用技能仍默认内核角色、指令数目标和设备 A/B。产品源码已最小修正四份 role SKILL：
research-discipline 按任务证据与目标研究并保留内核场景的五维结构审计；implementation-guardrails 服从批准范围与冻结策略；
validation-flight-check 区分 correctness、performance 和选定设备配方；review-checklists 按同一策略评审。
不再由通用方法强制本地 correctness 执行 IC、刷机、A/B、旧 singleton 或自动 Hub 发布。
这四份技能改动没有修改选定内核配方的方法与强制门禁；角色权限修复单独记录于第 4.1 节。

四份技能 quick_validate、LF 和 diff 检查通过；相关工作台、golden 配方、权限和 pipeline 测试 **57 passed，0.32 秒**。
这些测试验证资产与合同兼容，不证明修改后提示的自主执行效果。完整候选执行期间未替换 audit 副本的四份技能；本轮早已显式绑定的冻结 correctness 策略覆盖旧默认值。收尾权限探针前才复制最终资产。
reviewer/validator 上层角色也已按冻结 correctness/performance 策略修正；完整多角色流程尚未使用最终这些提示从头重放。

## 5. 本轮 correctness 候选与原生沉淀结果

最初快照中的 1 个候选现已由服务记录为 `validated` v6。基线为 `609087fd9f8950f02778c1ba8788954153544967`，
新提交为 `d813a4c00076875b5fc7e2ed6228bf8f822c9127`，实际 Git diff 仅有 `counter.py` 一行替换；
implementation digest 为 `a33eba3d10790b6a9507c767ae483430d16704268d4167d07b0839d0b7e52fe3`。
初始研究/方案 dispatch 已保存 approve_plan 接受回执；v3 的实施 dispatch `dispatch-ec887b31e09a662fdddc3817` 已完成并记录 v4。
v4 代码评审 dispatch `dispatch-31a0a31df99dba92b565b381` 已通过评审门，validator 使用 v5 的 `dispatch-f8e0c43a9e02207b7d1aade2` 执行并提交真实结果。
验证报告证据为 `95360252681cc60d85be77eac18117127ba790cb225fbf3fa093bb8940779a12`：kind=correctness、execution_kind=local、simulation=false、verdict=pass。
本次没有硬件指标：hardware_verified=false、metrics 为空、pairs=0。按新阶段/版本生成 dispatch 是正常权限交接，不是休眠恢复时重复执行同一请求。

| 阶段 | 本次判断 | 实际证据与边界 |
|---|---|---|
| 本地规格与基线复现 | 已有实际夹具执行证据 | 冻结规格、工作负载与环境文件；基线失败的原始输出及摘要 |
| 来源导入、策展、发现 | 已通过此隔离用例 | 真正调用的工具、来源、pattern 版本、覆盖与候选记录 |
| 责任人确认 | 已通过本地标签合同 | 已保存 confirm 决定、理由、版本与作用范围 |
| dispatch 与 researcher 委派 | 已有 dispatch；researcher 已实际完成 | 子会话 `ses_f72cc1140ffeN0TVtp2Tr5Fr4K`、research-note.md 与 execution attempts；身份/独立性边界见第 4.2 节 |
| 方案草案 | architect 已实际完成，后续已获方案门批准 | 子会话 `ses_f6f57aa49ffe4Le6g07VNjtnhx`；plan.json、plan.md、decisions.md 真实 write 回执 |
| 独立方案评审 | review 与 approve_plan 服务门已通过 | author 与方案作者不同、规范摘要匹配，接受后进入 v3/plan_approved；身份边界仍见第 4.2 节 |
| 实现与提交 | 真实提交且服务已接受为 v4/implemented | 新提交 d813a4c、仅 counter.py 的 diff、recorded-implementation.json；前置方案评审与后续代码评审、验证分别留证 |
| 独立代码评审 | review 与 approve_code 服务门已通过 | 对精确实施摘要的独立 reviewer 记录，接受后到 v5/code_approved；身份边界见第 4.2 节 |
| 本轮 correctness 验证 | 真实通过，服务为 v6/validated | 冻结六项在新提交全部通过，基线 whitespace_to_zero 失败被核验；保存原始输出与报告，非模拟、本地执行 |
| 经验捕获与原生沉淀 | 自动结果捕获、本地策展与原生 L1 staging 完成 | audit-curator CLI 回执、原生 schema 与来源解析、同 request 幂等检查；没有发布共享 Hub |
| 真机性能优化 | **未纳入本轮** | 必须另做真实内核、构建/镜像、设备、原始 A/B 和收益验收 |

本轮 baseline 五项通过、一项失败；候选六项全部通过，非法文字 ValueError 语义保持。
该结果只证明这一类受监督流程和这个局部修复的可执行性，不能推导出性能改善或大规模候选的高精度。

### 5.1 主机休眠后的恢复边界

主机休眠后，researcher 曾等待模型桥回复；根任务核对已有会话和请求，向最新重试 `1789076510235872700-06ea1571` 提供回复，
忽略已失效的旧请求 `1789075488156632800-eaff3de0`。随后原 researcher 实际完成，coordinator 继续委派 architect 和新的 reviewer。
恢复过程中复用既有 dispatch `dispatch-032ace04218b2c1000068535`，没有创建第二个 dispatch 或重复 owner 确认。
`execution.json` 的 recovery_note 明确记录保留原 dispatch、没有重放服务门。

这些回执证明本次人工监督下的模型传输恢复成功，不能据此声称无人值守 provider 的超时、断连和重试已经全面通过验收。
模型传输失败仍需与候选业务失败区分。后续恢复应先读已有 session、dispatch、execution 与候选版本，
再对未完成请求续接或重试，避免因页面等待而重复创建任务、确认或重放已完成的写操作。

validator 阶段另发生过服务进程退出，原因尚未确定。`validator-process-recovery.json` 记录服务与模型桥端口关闭后，
使用相同持久 XDG 会话存储重启，并向原 validator 会话 `ses_f6f36952effeM4FR9jkIFRUhaY` 发送恢复请求。
该次恢复要求保留原 dispatch、已完成工具结果与审批，不重做确认、实施或评审；同样属于人工传输/进程恢复。
验证结束后，主 coordinator 会话也通过显式恢复消息读取最终服务状态与工件，然后正常返回 `finish=stop`；没有重放业务门。
最初 full 命令的 HTTP 调用者曾超时，未取得同步响应文件；最终结论来自持久会话 API 和服务回执，不将调用者超时写成一次无中断成功。

### 5.2 自动化测试记录与中断影响

以下软件回归与上述真实 OpenCode 候选验收分别记录，重跑/重叠用例不相加为一次全量结果：

| 运行范围 | 实际结果 | 判断 |
|---|---|---|
| 全量测试 | 1097 passed、6 skipped、2 errors；耗时 16:01:43，跨主机休眠 | 两项为超时错误；这是带错误的全量运行，不能标为一次完整 clean pass，也不能把休眠跨度当作正常运行性能 |
| 重跑 `tests/test_evolution_quality.py` 全文件 | 53 passed，60.32 秒；包含原来两项错误用例且均通过 | 证明两项在本次重跑已通过；不抹掉前次错误记录，不与前次通过数相加为一次全量结果 |
| 四份通用角色技能修正后的工作台合同回归 | 57 passed，0.32 秒；技能 quick_validate 通过 | 检查 Evolution 工作台、golden 配方、权限和 pipeline 兼容，非提示行为评测 |
| 后续权限收窄及上层角色修正回归 | 81 passed，6.06 秒；Ruff/diff/LF 检查通过 | `final-test-verification.json`；包含副作用命令回归，不代表整个最终提示已从头运行 |

软件测试与本地流程通过均不能替代设备 A/B、生产规模覆盖或真实优化收益证据。

### 5.3 原生 journal 与 L1 staging 的实际验收

只在候选真实 validated/pass、simulation=false 后执行 curator 操作。自动捕获记录 `skill-486c277b35a38c4ff758d20c`
由 audit-curator 提升为 staging，再以 contributor=audit-owner、project=correctness-audit 导出明确审阅过的局部事实。
`native-curation.json` 保存全部真实 CLI 参数、退出码和结果；导出 ID 为 `native-export-5562632bf7ddffe9e8c3f1b57c5621df`。

原生 memory_item schema 校验无错误，3 个 Evolution 证据引用经 `resolve-native-evidence` 实际解析。
相同 request 再导出返回完全相同结果，两个根目录的文件 hash 不变；最终只有 1 份 journal（`J-01M28E5P8644PTE427MWAM4N6C`）和 1 份 staging。
原生 `central_curate.py --plan` 只读接收该暂存项并给出 add 计划，没有执行 apply，也没有生成 knowledge 或 skills。
最终状态为 **L1 / staged / not_published / merged=false**；这是本地可审阅沉淀，不是共享 Hub 发布、独立复现或新 pattern 的自动再消费。

## 6. 配置和工作台使用体验

### 6.1 已降低的配置成本

[三步快速开始](EVOLUTION_QUICKSTART_CN.md) 已把首选路径收敛为：明确 repo/owner → setup 生成单文件配置与 fragment → doctor → 工作台两个入口。
统一主 MCP 接入避免再维护独立 Evolution 连接，状态、工作区、工件和 pilot profile 由一份配置承载。
setup/doctor 默认提供中文简报，`--json` 供脚本使用；provider 与既有连接不被 setup 自动重写。

本次 runtime 证明主 MCP 实际可列出并执行 Evolution 工具；doctor 本身仍只证明本地配置/组件条件。
远程与容器环境仍需要把配置文件及其目录映射为服务端可见路径，并明确检查非空高级环境变量覆盖，不能把本机 setup 成功当作远程已接通。

### 6.2 人工门仍需要切到 CLI

当前交互在草案策展和候选确认处会暂停，操作者分别执行 import/activate 与 decide confirm，再回到 OpenCode。
本次正是通过 `operator-curation.json` 和 `operator-confirmation.json` 中的命令完成，而非直接回复“同意”改变状态。
版本、候选 ID、actor 和请求 ID 有助于可审计，但对普通专家而言步骤仍重。

建议产品入口提供候选摘要、证据链接和带实际参数的确认操作说明，优先减少手工抄写长 ID；
若引入网页/消息确认，须实现真实身份、版本与回执校验，不能把聊天文本交给 Agent 猜测为授权。
目前没有通讯录、邮件/消息投递、回复回调或自动会话唤醒，不能声称只填邮箱即可启用。

### 6.3 命令模板对用户消息的可读性

实际 `session-messages.json` 的用户消息包含展开后的 discovery recipe、角色引用、技能读取结果和附加说明，而不仅是简短的 `/evolve-discover pilot`。
根任务随后通过 CUA 实际检查 tab 4 的可访问性树，确认已有会话可打开，页面包含完整用户命令、长模板与当前 Thinking 状态。
因此“消息展开过长”的判断已有页面层观察支持，已超出 served bundle 路由分析和 HTTP 200。
这会增加会话中可见文本量，也使“用户请求”“加载的执行合同”“工具结果”更难快速区分；这些文本仍是被审阅内容，不是新的操作指令。

建议把操作合同保留在技能/工件中，在用户界面优先显示短命令、当前阶段、下一人工门与证据入口；需用实际 OpenCode 的可用机制验证，不能仅删掉必要合同来缩短文字。
后续应继续验收首次进入、确认暂停点、错误恢复和最终阶段展示；本次 CUA 可读与状态检查不覆盖所有按钮、输入和交互分支。
收尾时再次尝试 CUA 窗口读取，native pipe 返回 unavailable / failed connect（os error 2），因此停止 CUA 操作。
先前 tab 4 的实际观察仍有效；本轮最终执行结果只能另由持久 OpenCode 会话 API 和服务回执确认，不能声称最终页面已完成复验。

### 6.4 独立目标仓库的权限交互成本

本轮目标与工作台分开存放，实际 `read` 对规格、源码和冻结证据触发多次 `external_directory` 确认。
recipe 要求显式 `git -C <target> ...`，但本轮运行副本中旧的 `git status*`、`git diff*` 白名单不匹配前置 `-C`，
实际 HEAD/status 查询也触发了 bash 确认。本轮逐条检查后仅批准当前操作，回执保存在 `permission-decisions.jsonl`。

这些是运行成本，不能以“只需两条命令”概括完整体验。后续应设计绑定目标仓库的最小只读授权和清楚的执行目录提示，
在明确目标与动作后减少重复确认；源码修改、提交和设备操作仍按各自权限处理。不要用全局 bash 放开来掩盖交互问题。

## 7. 与最初通用优化目标仍有的差距

下表延续[独立规划矩阵](EVOLUTION_PLANNING_ALIGNMENT_REVIEW_CN.md)，本轮小型 runtime 用例没有消除这些平台能力缺口。

| 能力 | 本轮证明了什么 | 仍需交付或验收 |
|---|---|---|
| 规模化历史与全库扫描 | 小仓库的历史、来源和批次接入实际可运行 | 并行/batch Git、可继续的文件分区/游标、完整覆盖汇总；5 万提交或 2 万 C 文件级别的耗时尚未实测 |
| pattern/反模式蒸馏 | 关键词与字面片段能产草案，人工策展规则可命中 | 问题/修复/反例判别、跨记录聚合、结构锚点、语义 judge 与稳定支持证据 |
| 热点与瓶颈 | 未配置热点时明确显示缺口，score 不冒充概率 | 常态化性能摄取、符号级热点/调用图关联、瓶颈分类、趋势/回归区间定位 |
| 候选质量 | 一个有明确规格的隔离诊断候选成立 | 独立标签集、各漏斗阶段误报/漏报、收益/成本排序及真实专家裁决数据 |
| 正负反馈与自演进 | 确认、验证结果捕获、成功事实本地策展和来源解析已运行 | 自动质量策略仍缺；native 出口主要接受成功 fact，失败/拒绝共享与新 pattern 再消费未在本轮验证 |
| Skill Hub 共享 | 原生 schema、个人 journal、L1 staging 与只读 curator 计划已运行 | 可执行 pattern 的评测/发布/跨任务消费，不能由本地暂存或内部 tier=hub 替代 |
| 批量调度与专家协作 | 显式工作台入口可分阶段调用 | 目录、可靠通知、认证审批、scope 绑定续跑、周期批次与跨成员去重 |
| 执行与测量可信性 | 有实际 runtime/MCP/write 和本地检查回执 | 独立执行者身份、角色隔离、采集器来源关联及真实设备结果；hash 和标签不提供身份认证 |

## 8. 交付记录与后续验收

1. 最终主会话与六个子会话已保存；主会话正常返回，服务为 validated v6，native staging 可解析且未发布。
2. 权限顺序、命令范围、通用目标修复与报告整合到同一个功能提交；远程同步以显式 lease 防止覆盖其他人的新提交，结果见最终交付回执。
3. 本地证据包固定所验收版本、会话、工具执行、最终状态与必要原始工件；最终提示已加载并验证权限，完整多角色提示重放仍需单独验收。
4. 区分本次监督者提供模型响应和未来正式 provider 自主运行；后者需要独立连接与执行验收，不能复用本次结论。
5. 保留 owner/curator、人机切换和长模板界面的体验问题，给出可复制的最短操作路径。
6. 对真实内核与真机另设试点验收，包含功能回归、配对测量、冻结指标、噪声与收益，不以 correctness 或协议通过代替。

## 9. 本机证据索引

以下文件均相对本报告中的 `AUDIT`，可用于根任务最终归档和复核：

| 文件/入口 | 内容 |
|---|---|
| `session.json`、`session-v2.json` | 主会话身份与 runtime 元数据 |
| `session-messages.json` | 早期命令展开、MCP 调用与会话快照 |
| `final-session-messages.json`、`final-session-children.json`、`final-role-sessions/` | 已完成主会话及六个子会话完整回执，包含实际最终返回 |
| `final-candidate.json`、`final-candidate-audit.json`、`final-skill.json` | v6 候选、业务审计与策展后经验记录 |
| `coordinator-final-recovery.json`、`candidate-full-request-timeout.json` | 最终主会话人工续接与原同步调用者超时记录 |
| `runtime-agents-final.json`、`final-runtime-permission-check.json`、`final-permission-probe-messages.json` | 最终资产实际加载、13 个定义核对与写入/审批探针 |
| `final-git-push.json` | 最终 amend 提交、预期远程 lease、推送结果与远程核对值 |
| `command-missing-profile.json`、`command-discover.json` | 缺 profile 停止与首轮 draft-only 发现 |
| `operator-curation.json`、`discovery-after-curation.json` | 显式策展回执与已激活规则命中 |
| `candidate-before-confirm.json`、`operator-confirmation.json` | 未确认阻断及 owner CLI 明确确认 |
| `recorded-implementation.json` | v4/implemented 实际服务记录，包含获批方案、评审、基线/新提交与实现摘要 |
| `implementation-dispatch.json`、`code-review-dispatch.json` | v3 实施与 v4 代码评审的分阶段工作区和绑定 |
| `test-verification.json` | 全量测试中的两项超时与受影响文件重跑结果，分别记录 |
| `final-test-verification.json` | 后续权限/角色修正的 81 项通过记录与相关静态检查 |
| `validation-dispatch.json`、`validator-process-recovery.json` | v5 validator 绑定与相同持久会话的人工进程恢复 |
| `workbench/.opencode/local/workspaces/evolution-dispatch-f8e0c43a9e02207b7d1aade2/artifacts/validator/` | 新提交实际执行、原始 stdout/stderr、逐项证据、冻结报告与验证回执 |
| `native-reviewed-content.json`、`native-curation.json` | 明确审阅过的局部事实、真实 CLI 策展/导出/重放/解析与只读 curator 回执 |
| `native-memory/audit-owner/correctness-audit/journal/`、`native-hub/staging/audit-owner/` | 1 份原生个人 journal 与 1 份 L1 暂存候选，没有共享发布 |
| `permission-session-messages.json` | 修复前写工具不可见的实际反馈 |
| `permission-after-session.json`、`runtime-agents-after-permission-fix.json` | 修复后会话身份与 runtime 角色配置 |
| `workbench/.opencode/local/workspaces/permission-audit/probe.txt` | 实际 write 工具写入的测试工件 |
| `correctness-preparation.json`、`correctness-fixture/` | 规格、历史、基线、检查命令、原始输出及摘要 |
| `workbench/.opencode/local/evolution/state/` | 实际候选状态、审计与 dispatch；读取时须考虑并发更新 |
| `workbench/.opencode/local/workspaces/evolution-dispatch-032ace04218b2c1000068535/` | 本轮候选工作区与执行状态 |
| `model_bridge.py`、`model-queue/` | 受监督请求/回复桥及传输回执；不含外部 provider 性能证明 |
| `GET /session/ses_f72e2b23bffemLnAH461iNuoyp/message` | 已保存至 final-session-messages.json 的主会话记录 |
| `GET /session/ses_f72d429b7ffebSLmgddpSIbU8G/message` | 修复后 write/completed 的实际回执 |
| `GET /session/ses_f6f4813deffeUi8UCjo4vdc1vC/message` | 实施者的真实修改、Git 身份失败及单命令测试身份重试回执 |

本地归档位置为 `.opencode/local/audits/evolution-20260911/acceptance-evidence.zip` 与同目录 `archive-receipt.json`，
归档回执记录压缩包 SHA-256，包内 manifest 逐文件记录路径和摘要。白名单包含必要收据、原始夹具、阶段工件、原生 journal/staging；排除模型队列、provider 配置、可执行文件、live SQLite 和 Git objects。

本报告不汇总“全流程通过率”或“总收益”：实际证据支持这个受监督本地 correctness 案例及原生暂存闭环，生产试点与最终提示完整重放仍须另验收。
