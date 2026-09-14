# Evolution：三步开始使用

业务根目录包含多个 Git 仓库时，使用 [多仓配置与运行指南](EVOLUTION_MULTI_REPO_RUNBOOK_CN.md)：
一次 `setup-workspace` 选择关注项目，再用 `/evolve-workspace` 和 `serve` 推进。
下面保留单仓试点的最短入口。

仅需维护原平台 YAML；仓库范围、通知和验证的必要字段见[实际运行配置](../configs/evolution/README.md)。
发现 profile 自动生成，验证策略和结果随任务归档，无需复制独立 JSON 配置。

在已有 OpenCode 工作台中，复用配置好的仓库路径，首次只需**明确的试点负责人**。
日常使用两个工作台入口：发现候选、执行已确认候选。

## 1. 在原平台配置中登记范围

确保原 `.env` 已通过已有启动方式进入进程环境；Python 不会自动加载任意 `.env.xx` 文件。
在工作台项目根目录执行，把 `alice` 换成真实负责人：

```bash
python -m pip install -e '.[validation]'
python -m hmopt evolve setup --owner alice
```

仓库继承原 `KERNEL_REPO_PATH` 或平台 `project.repo_path`，也可用 `--repo /path/to/kernel` 覆盖。
不在工作台目录时，追加 `--workbench /path/to/workbench`。Windows 使用 Windows 绝对路径。
setup 在 `HMOPT_MCP_CONFIG` 指向的原平台 YAML（默认 `configs/app.yaml`）追加 `evolution` 节，
登记名为 `kernel` 的发现 profile，并生成连接片段。状态/工件/任务路径自动继承，状态库按需创建。
setup 不启动发现，也不修改 provider 或现有 MCP 连接。

## 2. 配入已有主 MCP，检查配置

主 MCP 继续使用原平台配置；只有原连接尚未指定配置路径时，按生成片段设置已有变量：

```json
"HMOPT_MCP_CONFIG": "/path/to/workbench/configs/app.yaml"
```

本地主 MCP 使用 `python -m hmopt.api.mcp_stdio`；远程主 MCP 则把变量配在服务进程侧，沿用原 `/mcp` 地址。
远程或容器部署时，配置文件及其仓库、工作台和工件路径都必须在服务端可见；本机配置不会自动上传。
只需接入一次并重启主 MCP，保留现有 provider 和其他配置。然后在工作台根目录执行：

```bash
python -m hmopt evolve doctor
```

doctor 按当前工作台的配置检查 Git、工具和组件就绪情况；依提示修复缺项。
指定另一份平台配置时用 `python -m hmopt evolve --config /absolute/configs/app.yaml doctor`。
`setup` 和 `doctor` 默认输出中文简报；脚本调用时在命令末尾加 `--json`，例如 `python -m hmopt evolve doctor --json`。
若提示旧环境变量覆盖配置（JSON 字段 `environment_overrides`），先核对这些变量；它们会覆盖文件中的对应值。
旧 JSON 仍兼容。仅在工作台没有平台 YAML 时 setup 才保留旧独立 JSON / `pilot` 模式；
旧部署继续使用原 profile 名称，迁移需保留状态目录和项目身份。完整继承规则见[多仓指南](EVOLUTION_MULTI_REPO_RUNBOOK_CN.md#1-最短配置路径)。

## 3. 在工作台发现，再执行

```text
/evolve-discover kernel
```

工作台使用现有模型逐个分析历史提交的实际前后代码，提交说明只作为辅助；不需要另配挖掘模型。
每次最多处理五个提交，未完成时按返回的同一批次 ID 续跑；缺少代码上下文的项会明确显示为待补充。
模型输出“改了什么、行为如何变化、原因推断、适用条件与反例”，后端校验代码引用和搜索规则后生成草案。
这是有界交互模式，尚不是自动并发的后台 Agent 调度。
完整方法、Python/Agent 执行方式比较与当前边界见[历史挖掘设计](EVOLUTION_HISTORY_MINING_DESIGN_CN.md)。

工作台展示批次、覆盖率、候选证据和下一步。首次运行可能只有 pattern 草案：
先由策展人核对并激活，再筛选；新历史模式的候选须先完成语义适用性复核，再由责任人明确确认。

```text
/evolve-research kernel synthesize <source-id-1> <source-id-2>
/evolve-research kernel review <pattern-id@version>
/evolve-queue kernel pending
/evolve-research kernel assess <candidate-id>
```

前两步用于跨提交归并：至少两份已完成历史分析，形成草案后独立评审，之后仍由策展人决定激活。
单提交草案也可按原流程策展。`assess` 按机制、每项前提和反例排除条件引用固定版本源码，
存在未知条件就保持待研究。新复核会更新候选版本，确认前应刷新候选/确认单。
工作台会给出带实际候选 ID、责任人和版本的 CLI 确认命令，**由责任人填写理由并执行**。
确认成功后，在工作台输入：

```text
/evolve-candidate <candidate-id> full
```

它从当前允许的阶段推进研究、方案评审、实现、代码评审与验证；遇缺失证据或人工门停止。
同一套配置还可查看已确认队列，显式选择最多十项逐个执行：

```text
/evolve-queue kernel approved
/evolve-batch kernel full <candidate-id-1> <candidate-id-2>
/evolve-batch resume <execution-batch-id>
```

每个候选使用批准基线的独立 Git worktree；复用已有角色和模型，不增加后台模型配置。
批次保留方法快照、审批 ID、进度和阻塞原因，不自动合并或推送代码。
恢复前工作台须确认原子会话已停止，不能对仍在运行的候选重复启动。
完整方案、接口映射和验收范围见[Skill 工作流设计与实施](EVOLUTION_SKILL_WORKFLOW_IMPLEMENTATION_CN.md)。
计划使用 lmbench 时，须在方案评审前[冻结指标 profile](EVOLUTION_OPENCODE_RUNBOOK_CN.md#6-在方案评审前冻结-lmbench-指标)。

## 谁确认，确认后怎样继续

`--owner alice` 表示你明确指定 alice 承接此次仓库试点范围，生成的默认规则是 `{"**": "alice"}`。
这不是自动识别专家，也不是邮箱或企业账号。扩大使用范围前，按源码路径细化配置中的责任规则。
上述最小配置使用 CLI/确认单完成确认，未启用外部通知。需要专家通知时，在同一配置中启用
`production.approval`，接入责任目录、SMTP TLS/webhook 与认证网关；网关回调可原子保存批准及续跑记录。
确认后仍显式选择候选执行范围，不自动唤醒优化会话。完整接入见 [生产使用指南](EVOLUTION_PRODUCTION_RUNBOOK_CN.md)。
详见[人工确认操作](EVOLUTION_OPENCODE_RUNBOOK_CN.md#5-责任人确认与启动)和[审批桥设计](EVOLUTION_APPROVAL_BRIDGE_DESIGN_CN.md)。

## 只执行一个步骤

| 目的 | 工作台输入 |
|---|---|
| 查看候选状态 | `/evolve-candidate <id> status` |
| 只推进当前阶段 | `/evolve-candidate <id> stage` |
| 只研究 | `/evolve-candidate <id> research` |
| 研究、方案和独立方案评审 | `/evolve-candidate <id> plan` |
| 只实施 | `/evolve-candidate <id> implement` |
| 只代码评审 | `/evolve-candidate <id> review` |
| 只验证 | `/evolve-candidate <id> validate` |
| 续跑未完成发现批次 | `/evolve-discover kernel <返回的批次ID>` |

单步也需要满足前置状态。独立 mine/sources/distill/scan 工具见[完整工具表](EVOLUTION_OPENCODE_RUNBOOK_CN.md#22-mcp-工具与可调用步骤)。

## 怎样判断它已经有效

| 层次 | 能说明什么 |
|---|---|
| 配置与预检查通过 | 路径、工具和相关组件满足检查条件；doctor 不产生候选，不运行模型或设备 |
| 候选通过证据门 | 本次候选具备规定的确认、独立评审、实现和验证证据 |
| 实测生产收益 | 在真实目标与设备上取得可复核的 A/B 结果，并满足预先冻结的指标与回归门槛 |

doctor 通过不等于模型会话已跑通，也不代表已经取得优化收益。
多仓库、外置工件目录、真机数据、故障恢复与 Skill Hub 沉淀见[进阶操作指南](EVOLUTION_OPENCODE_RUNBOOK_CN.md)。

生产后台挖掘、专家通知与认证回调、质量评测的配置及使用见 [生产增量指南](EVOLUTION_PRODUCTION_RUNBOOK_CN.md)。
