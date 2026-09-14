# Evolution：实际运行配置

只维护原 `HMOPT_MCP_CONFIG` 指向的平台 YAML：通常是 `configs/app.yaml`，
Docker 使用 `configs/app.docker.yaml`。本目录只保留这份说明，无需复制或加载额外 JSON。
原 `.env` 通过已有启动方式进入进程环境；模型凭据继续由原 OpenCode 配置管理。

## 1. 必需：关注的仓库与责任人

在 HMOPT 工作台根目录执行一次，替换实际项目路径和负责人：

```bash
python -m hmopt evolve setup-workspace --project kernel --project memory=foundation/memory --owner actual-owner
python -m hmopt evolve doctor
```

setup 在原 YAML 追加 `evolution` 节，保留其他配置和注释。已有此节时直接编辑；
单仓试点可用 `setup --owner actual-owner`。生成的 MCP 连接片段仅供接入已有主 MCP，
其配置仍指向同一平台 YAML。远程 MCP 需在服务端配置并重启，片段不会自动上传本地目录。

手动编辑时，最小多仓配置如下；替换负责人，按需保留项目：

```yaml
evolution:
  source_workspace:
    workspace_id: business
    projects:
      kernel:
        owners: {"**": kernel-owner}
      memory:
        path: foundation/memory
        owners: {"**": memory-owner}
    selected: [kernel, memory]
```

- `workspace_id` 与项目名用于归档身份，已有运行后保持稳定。
- `selected` 是明确关注范围；未选择仓库不参与挖掘和扫描。
- `owners` 是仓库相对路径 glob → 责任人标识，可按目录细分；标识不等同于企业账号或邮箱。
- 仅参与构建的仓库也登记在 `projects`，在 `source_workspace.dependencies` 列出其项目名，
  不加入 `selected`。框架冻结其版本，不挖掘或扫描。

业务根复用 `PROJECT_REPO_PATH`（兼容 `KERNEL_WORKSPACE_PATH`）；kernel 路径复用
`KERNEL_REPO_PATH` 或平台 `project.repo_path`。其他项目 `path` 是业务根下的 Git checkout 相对路径。
仅在原配置缺失时使用 setup 的 `--source-root`、`--repo` 或显式 `--project 名称=相对路径`。

## 2. 按需：后台研究、专家通知、业务验证

以下片段都合并进**已有 `evolution` 节**；启用多个能力时合并到同一个 `production`，
不要创建重复 YAML 键。不使用的能力不添加配置。

后台批量研究：

```yaml
production:
  worker: {}
```

目录、researcher/顶层模型与已配置本地 `server.port` 从工作台或显式 OpenCode 配置继承。
动态端口或远程服务才补 `worker.url`；无法读取模型时补 `provider_id` 和 `model_id`。
`url` 是 OpenCode session 服务地址；需鉴权时 `authorization_env` 引用已有环境变量。
交互式工作台研究无需此 worker 配置；后台任务用 `python -m hmopt evolve serve` 推进。

专家通知与认证审批，以 webhook 为例：

```yaml
production:
  approval:
    contacts:
      kernel-owner:
        principal: "your-tenant:actual-expert-id"
        target: "actual-message-target"
    gateway_url: https://approval.example.invalid/evolution
    signing_key_env: HMOPT_APPROVAL_GATEWAY_KEY
    webhook_url: https://notification.example.invalid/evolution
    webhook_authorization_env: HMOPT_NOTIFICATION_AUTHORIZATION
```

为需要通知的每个 owner 登记实际联系人，替换网关和通知地址。
`signing_key_env` 指向至少 32 字符的网关共享密钥；通知渠道无需鉴权时可省略
`webhook_authorization_env`。这里填变量名，密钥值从部署环境注入。
使用邮件时删除两个 webhook 字段，改填 `smtp_host`、`smtp_sender`；需要登录时
同时提供 `smtp_username`、`smtp_password_env`，默认隐式 TLS 端口 465。
企业 SMTP 使用 STARTTLS 时设置 `smtp_security: starttls`、`smtp_port: 587`；
协商失败即停止，不降级明文。Docker Compose 已透传标准密钥变量
`HMOPT_APPROVAL_GATEWAY_KEY`、`HMOPT_SMTP_PASSWORD`；若环境引用采用其他名称，
需要在部署环境中传入对应变量。
通知投递、企业登录与回调接入见[生产指南](../../docs/EVOLUTION_PRODUCTION_RUNBOOK_CN.md)。
未启用通知时，沿用工作台展示的责任人 CLI/确认单流程。

自动构建和验证：

```yaml
production:
  validation:
    command: ["/absolute/python", "/absolute/your_business_adapter.py"]
```

替换为实际可执行的适配器命令；构建、刷机、测试配置继续继承原环境。
适配器须执行批准的基线和候选测试并返回真实结果，接入协议见
[业务验证](../../docs/EVOLUTION_MULTI_REPO_RUNBOOK_CN.md#4-接入已有构建和真机脚本)。
默认工作目录为业务根，同一状态库中的验证串行占用共享资源；独立设备才按需覆盖 `resource_id`。
适配器未接入时仍可由工作台组织验证、提交实际证据。

## 3. 无需重复配置的内容

| 内容 | 来源 |
|---|---|
| MCP 入口、平台配置路径 | 现有主 MCP 与 `HMOPT_MCP_CONFIG` |
| Git 仓库发现 profile | 根据 `source_workspace.selected` 自动生成 |
| 工件目录 | 原平台 `storage.artifacts.root_dir` / `storage.artifacts_root` |
| 工作流状态目录 | 工件目录旁的 `evolution/`，独立状态库 |
| Agent 任务目录 | 工作台 `.opencode/local/workspaces/` |
| Git、分页、并发、预算和超时 | 代码默认值，仅在实测需要时覆盖 |
| 正确性检查、指标与门槛 | 每项任务的方案，由独立评审冻结 |
| A/B manifest、版本、摘要与结果 | 采集适配器或验证流程生成，随任务归档 |

验证数据协议以 `python -m hmopt evolve schema <类型>` 为准，类型包括
`plan`、`correctness-policy`、`correctness-report`、`lmbench-profile`、`lmbench-manifest`。
它输出字段定义，不生成实际证据；采集器必须从真实运行取得版本、摘要和结果。
lmbench 指标选择及原始文件导入见[验证步骤](../../docs/EVOLUTION_OPENCODE_RUNBOOK_CN.md#6-在方案评审前冻结-lmbench-指标)。

修改平台配置后重启原 MCP，运行 `doctor` 检查；后台 worker 变更按生产指南处理活动任务。
旧 JSON 与 `HMOPT_EVOLUTION_*` 仍可读取，已有部署无需立即迁移；新接入使用同一平台 YAML。
旧变量覆盖文件值时 doctor 会报告；迁移先保留状态/工件路径与项目身份，再清除旧覆盖。
本次示例清理不删除已有运行数据或审批归档。
