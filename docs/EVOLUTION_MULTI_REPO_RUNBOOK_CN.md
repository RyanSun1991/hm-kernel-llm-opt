# 多 Git 业务工作区：配置、运行和验收

一个大型 repo 工作区可以包含多个独立 Git 仓库，根目录不需要是 Git 仓库。
本指南适用于 repo 工具管理的目录及普通多 checkout 目录；不会执行 repo sync、扫描未选择仓库或自动修改源码。

## 1. 最短配置路径

沿用原 `.env` 中的 `PROJECT_REPO_PATH`（业务 repo 根）、`KERNEL_REPO_PATH`（内核 checkout）和
`HMOPT_MCP_CONFIG`（通常为 `configs/app.yaml`；Docker 为 `configs/app.docker.yaml`）。
这些变量必须已进入启动进程的环境；Docker 沿用原 env-file，直接运行 Python 时沿用已有环境加载方式。
在 HMOPT 工作台根目录、已安装项目的 Python 环境执行一次：

```bash
python -m hmopt evolve setup-workspace --project kernel --project memory=foundation/memory --owner pilot-owner
```

只需把新增项目相对路径和负责人换成实际值；`kernel` 路径继承已有配置，工作区 ID 默认 `business`。
不在工作台目录时追加 `--workbench /srv/hmopt`。未配置业务根目录时可追加 `--source-root /srv/business-repo`；
也可用 `--project kernel=kernel/main` 显式指定内核位置。Windows 同样可用，绝对路径使用 `C:/...`。
项目 path 是相对业务工作区的路径。
同一个 Git 仓库可以使用正常 `.git` 目录、Git worktree 的 `.git` 文件或 repo 管理的 Git 元数据。
注册的项目目录必须是准确的 checkout 根目录；路径不能越界或经过重定向。

setup 在原平台 YAML 中追加 `evolution` 配置节，并返回一个连接片段：

- 原 `configs/app.yaml` / `configs/app.docker.yaml`：按项目修改 `evolution.source_workspace.projects.*.owners`，即可分配不同责任人。
- `opencode.fragment.json`：现有主 MCP 继续指向同一 `HMOPT_MCP_CONFIG`，无需再维护一份 Evolution JSON 或新增环境变量。

若原主 MCP 是 remote，更新服务端原平台 YAML 并重启原连接；配置和业务目录都需在服务端可见。
此时不要再添加一个并行的本地主 MCP。生成片段用于说明连接设置，不会自动上传目录。

setup 保留原 YAML 的注释和字段；已有不同的 `evolution` 配置时提示直接编辑该节。
它不创建模型、启动挖掘或发送通知。

[实际运行配置](../configs/evolution/README.md) 集中说明仓库范围、通知和验证的必要字段，
全部写入同一平台 YAML，无需额外 JSON。需要后台批量挖掘时，在同一 `evolution` 节补充：

```yaml
  production:
    worker: {}
```

worker 默认使用 HMOPT 工作台目录，复用该目录的 `opencode.json` / `opencode.jsonc`、
显式 `OPENCODE_CONFIG` 和 `OPENCODE_CONFIG_CONTENT` 中的 researcher 模型或顶层模型。
模型密钥仍由 OpenCode 管理。若上述配置已有本地 `server.port`，连接地址也自动继承；
使用动态端口或远程服务时，仅补 `worker.url`。无法解析模型时补 `provider_id` / `model_id`，不会猜测。
`HMOPT_LLM_BASE_URL` 是模型网关，不能充当 OpenCode session 服务地址。
OpenCode server 需要支持本项目使用的 session/prompt_async/message API；已有工作台启动方式保持不变。

| 配置内容 | 默认来源 |
|---|---|
| 业务 repo 根 | 原 `PROJECT_REPO_PATH`，兼容 `KERNEL_WORKSPACE_PATH` |
| kernel checkout | 原 `KERNEL_REPO_PATH`，回退平台 `project.repo_path` |
| 验证工件目录 | 原平台 `storage.artifacts.root_dir` / `storage.artifacts_root` |
| 工作流状态目录 | 工件目录旁的 `evolution/`，默认 `data/evolution/`，独立状态库 |
| Agent 任务目录 | 工作台 `.opencode/local/workspaces/` |
| Git、并发、超时、上下文预算 | 代码默认值，按确有需要显式覆盖 |
| 构建、刷机、测试参数及凭据 | 验证子进程继承原有环境变量 |

自动 A/B 验证只需补 `production.validation.command`（argv 列表，首项为可执行文件绝对路径）；
默认 cwd 是业务根目录，超时 3600 秒，同一状态库中的验证使用共享资源锁串行执行。
该命令须按下文协议执行基线/候选双侧测试，已有单侧 build 命令不能直接当作双侧验证结果。
有多台独立设备时再显式设置 `resource_id`。未接入适配器时仍可使用工作台手动验证入口。

旧独立 JSON 和 `HMOPT_EVOLUTION_*` 仍兼容，且显式旧变量优先覆盖；`doctor` 会列出覆盖项。
已有运行迁移时应保留原 `root` / `workspace_root` / `artifacts_root` 和项目身份，确认 CLI/MCP 指向同一份归档，
再停用旧变量；setup 不迁移状态库。相对平台路径以工作台根解析（`configs/` 的父目录）。

`selected` 指定需要挖掘和扫描的仓库。`dependencies` 可指定仅参与构建基线的仓库：
冻结它们的 revision，但不采集历史、不扫描、不产生优化任务。其他未选择、未声明依赖的仓库不读取。
若构建还依赖其他 Git 项目，应明确加入 dependencies；本平台不会假设未声明项目也已固定。

各 project 还可设置 `hotspots`，例如 `[{"path":"mm/reclaim.c","symbol":"reclaim_pages","weight":0.8,"revision":"实际40位提交SHA"}]`。
这是已有 profiler 数据的版本化输入，随总任务冻结并参与候选排序；不匹配目标版本时扫描拒绝使用。
未配置热点仍可进行代码模式筛选，不会编造热点或收益排序依据。本轮未实现自动持续采集 profiler。

## 2. 启动和使用

```bash
python -m hmopt evolve doctor
python -m hmopt evolve serve
```

serve 是一个可由现有服务管理器托管的前台进程，推进已显式创建的总任务，并轮询研究 Worker。
它不会自行创建新总任务或执行构建/设备操作。Ctrl+C 停止调度，远端模型状态仍归档可查。
一次调度检查可以使用 `serve --once`。
Windows 若执行现有 Hub Python 命令时遇到控制台编码错误，可先在 PowerShell 设置 `$env:PYTHONUTF8='1'`；
这只影响运行时编码，不转换源码换行。

在现有 OpenCode 工作台：

```text
/evolve-workspace start
/evolve-workspace status <返回的-run-id>
```

start 不带项目名时使用 config.selected；也可 `/evolve-workspace start kernel` 只研究一个已选择项目。
仅研究一个项目时，其他 selected 项目仍作为构建上下文冻结，但不启动本轮挖掘或扫描。
每个总任务冻结各仓库 SHA、项目逻辑 ID、路径、owner 及构建依赖，随后 HEAD 的变化不改变该总任务。
研究流程是：历史分页 → 代码分析/补充取证 → 已激活 pattern 的分区扫描 → 显示人工处理项。
各仓库有自己的检查点和错误，某一仓库阻塞不会停止其他仓库。

下一轮默认从同一逻辑项目及 revision 选择器的已采集 cursor 增量读取；上一轮未解决的分析会继续挂入新任务。
游标与项目检查点在同一事务中保存，采集完成不代表语义分析完成。需要重新遍历全部历史时，显式使用
`workspace-start --request-id <新ID> --full-history`，或 MCP start 的 `full_history=true`。
若分支重写导致旧 cursor 不再是目标祖先，任务进入 attention；检查后取消旧任务，再明确创建全历史任务。

状态中区分历史采集、已分析、needs_context、待处理、扫描覆盖及人工评审。
个别无法自动分析的提交不会被标成 no_pattern，也不会阻止其他有效 pattern 的扫描；它们保留在未解决列表。
`awaiting_review` 表示自动发现阶段已到人工处理边界，不能解读为所有历史都已准确分析或优化完成。

产出的 draft 需要 `/evolve-research <project> review <pattern-id@version>` 及策展人激活。
激活后执行 `/evolve-workspace scan <project>`，使用新 pattern 快照重新扫描。扫描返回 scan ID，
用 `evolution_scan(next, scan_id, expected_version)` 逐页续跑；工作台应保存 ID，不能反复 start 代替续跑。
已启动的 serve 也会推进这些扫描任务，工作台只需查看状态。

后续仍使用同一套入口：

```text
/evolve-queue kernel pending
/evolve-research kernel assess <candidate-id>
/evolve-production request-review <candidate-id>
/evolve-queue kernel approved
/evolve-candidate <approved-candidate-id> full
```

已确认的同仓候选可以 `/evolve-batch kernel full <id-1> <id-2>`。
一个候选的审批不覆盖其他 Git 仓库；跨仓修改需要分别形成候选、范围和审批。
不要把多个仓库的批准拼成未评审的整合补丁。

## 3. 后台调查与大文件

Worker 使用现有 OpenCode 模型和 evolution-mining Skill。模型可返回结构化 read/search 请求，
Python 只在该提交的 before/after Git 对象中读取文件窗口或进行字面搜索，再把归档结果交给同一会话。
模型不能通过该通道执行 shell、读取其他仓库/任意磁盘文件、任意更换 revision 或自己批准结果。

每轮保存消息 ID、原始模型输出和取证摘要。断线续查原消息；不确定请求不盲目重发。
默认最多 3 轮、每轮最多 4 个请求、累计补充上下文 128 KB。reported token 预算在轮间检查，
不是 provider 级硬 token 上限；未上报用量也不能声称成本已知。

文件正文较大但 diff 完整时，证据包保存全部变更 hunk 和真实行号映射；省略的未修改区域明确标记，
模型可补读定义、调用者和前后函数体。超大 diff、超过文件/总包预算、二进制或缺失证据仍保留未解决状态。
完整变更窗口不等于完整程序语义证明，引用和模式样例校验之后仍需独立复核。
升级前缓存的不完整证据包，可对指定 source 调用 `evolution_history_analysis(refresh=true, expected_version=当前版本)`
重新取证；旧证据保留，旧 Worker 的提交版本失效。只允许刷新未完成分析，不覆盖已完成研究。

扫描按固定 Git tree 和固定 pattern 版本分页，不会因为超过一次 max_files 就永远重复扫描开头。
每页候选全部归档；结果排名和待审核 Top-N 可以独立查看。跳过的大文件/二进制/特殊文件有计数；
二进制/编码跳过计数是各 pattern 分区的观测次数，同一文件可能重复计数，不是去重文件数。
`complete` 表示遍历结束，`coverage_complete` 才表示没有因这些原因跳过，不应混淆。
单个目录元数据超过 16 MiB 等极端边界仍会进入 attention。

## 4. 接入已有构建和真机脚本

`production.validation` 指定一个操作者管理的 argv 命令、工作目录、资源 ID 和超时。
同一资源 ID 同时最多一个实验；异常后不会仅凭超时释放可能仍忙碌的设备。
框架不替业务编写镜像构建和刷机命令；适配器调用已经落地的业务工具链。

代码评审通过后，validator 在 MCP 调用 `evolution_experiment(prepare, candidate_id, actor, request_id)`，
或操作员执行：

```bash
python -m hmopt evolve experiment-prepare <candidate-id> --request-id <唯一请求ID>
python -m hmopt evolve experiment-run <返回的-experiment-id>
```

prepare 不执行构建/设备操作；run 才执行指定的业务适配器，遵守已有角色和设备操作权限。
适配器可从环境变量读取：

| 变量 | 内容 |
|---|---|
| HMOPT_EXPERIMENT_REQUEST | 不可变请求 JSON 文件：候选、批准策略、两个 revision、执行 worktree、完整业务 manifest |
| HMOPT_EXPERIMENT_RESULT | 必须写入的 result.json 路径 |
| HMOPT_EXPERIMENT_ID | 当前实验 ID，用于外部流水线幂等关联 |
| HMOPT_TARGET_REPO | 当前候选的实际执行 Git worktree |

适配器等待构建/测试完成后退出，结果格式为 `{"report": <ABReport 或 CorrectnessReport>}`。
也可输出 `{"lmbench_manifest": <LmbenchManifest>}`，将原始 xlsx 放在本实验目录；
或输出 `{"ic_compare": <比较结果>, "ic_manifest": <ICManifest>}`。框架调用现有转换器保留原始来源，
再走同一验证门，不需要业务脚本直接写数据库。
不要把提交异步远端任务成功当作实验完成。原始日志可保留在本实验目录；stdout 不作为验收结果。
现有 lmbench/IC 转换器仍可使用，带 measurement_profile 的策略仍必须提供匹配的原始测量来源。

多仓候选的 plan 必须带 `workspace_manifest_sha256`。报告的每一侧必须提供 `project_revisions`：
键为 manifest 中 repo_id，值为该侧实际构建的 commit。baseline 与 manifest 完全一致；
feature 只替换本候选仓库为已评审的 implementation revision。其他关注仓库和 dependencies 不得变化。
报告没有列全、依赖漂移、目标版本错误、策略不符，都会拒绝接受。此协议依赖可信业务采集器，
不能当作加密硬件认证或证明适配器没有填写虚假元数据。

实验失败保留 `attention` 和资源占用。确认本地及远端执行均停止后：

```bash
python -m hmopt evolve experiment-retire <experiment-id> --version <当前版本> --execution-stopped
```

新的实验需明确 prepare；验证报告已产生 inconclusive 时，仍先走原有 owner retry_validation 门。

## 5. 通知、故障与续跑

配置生产审批桥的方法保持不变，见 [生产指南](EVOLUTION_PRODUCTION_RUNBOOK_CN.md)。
默认 serve 不发消息。需要它兼任通知 worker 时，显式启动 `serve --notifications`；
通知只投递已明确请求的审批对象。企业登录页面/网关仍由实际企业系统提供。

OpenCode 可用 `/evolve-workspace retry <run-id> <project-id>` 重试已修复的项目检查点。
历史预算默认每项目 10000 条，耗尽会停止采集并保留 cursor。明确扩展：

```bash
python -m hmopt evolve workspace-control <run-id> retry --project-id kernel --version <当前总任务版本> --max-commits 50000
```

新的预算必须大于原预算，不代表已经达到该规模吞吐。模型 attention/uncertain 仍按生产指南检查和 retire/retry。
取消总任务停止新调度，同时取消它关联的活动扫描，并使其活动研究结果失效；
扫描版本校验阻止旧页在取消后归档新候选。已归档候选保留，不等同于被拒绝。
远端计算可能尚未退出，容量不会被虚假释放。
重试/取消必须读取当前版本；后台有正在进行的项目步骤时会拒绝并发控制。

独立扫描失败进入 `attention` 时，修复原因后从原冻结 tree/pattern 检查点续跑：

```bash
python -m hmopt evolve show <scan-id> --kind scan
python -m hmopt evolve scan-control <scan-id> retry --version <当前扫描版本>
# 或停止这个扫描
python -m hmopt evolve scan-control <scan-id> cancel --version <当前扫描版本>
```

工作台对应 `/evolve-workspace scan-retry <scan-id>`、`scan-cancel <scan-id>`；
MCP 复用 `evolution_scan` 的 retry/cancel 动作，不增加新的服务。
重试不改变冻结版本、pattern 或覆盖范围；需要新范围时创建新扫描。
总任务中的失败扫描应优先使用项目 retry；若已单独取消其扫描，则取消总任务并重新启动，
不会把显式取消自动解释为允许恢复。

## 6. 沉淀和实际验收

通过/失败/专家拒绝继续进入原有 journal、质量统计和候选档案。策展人依据证据调整 pattern，
经独立评审激活新版本，再启动新 scan；实际通过的经验可以导出原生 Team Memory/Hub staging。
自动跨全历史语义聚类、AST/数据流证明、自动发布 Skill Hub 和跨机构分布式调度仍不属于已完成能力。

部署验收需要真实业务 repo 路径、现有模型、责任目录、审批网关和构建/设备适配器。
推荐先让两个目标仓库各完成一项从代码证据到真实验证的任务，再扩大历史和并发。
应保存真实分析、专家决定、patch、各角色评审、多仓基线和原始验证证据；收益和失败按实际结果报告。
