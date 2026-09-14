# Evolution 生产能力增量设计与实施

状态：本轮 P1–P4 已实现，生产协议与本地集成验证完成；真实环境部署验收见边界说明。

本文保留生产增量阶段的设计与测试记录。后续多仓、调查、分区扫描及业务实验适配器的当前实现见
[多仓实施说明](EVOLUTION_MULTI_REPO_IMPLEMENTATION_CN.md) 和 [运行指南](EVOLUTION_MULTI_REPO_RUNBOOK_CN.md)。

本轮目标是把已有的 Skill 推理、证据门禁与 ID 归档接到可持续运行的服务上。
Python 负责调度、认证、确定性校验和事务；语义推理由 OpenCode 已配置的模型执行。
所有能力复用同一个 Evolution 配置和 SQLite 状态目录，默认不开启外部投递。

## 1. 实施顺序与验收

| 阶段 | 交付 | 关键验收 |
|---|---|---|
| P1 | 持久化历史挖掘 campaign、OpenCode 后台 worker | 固定输入/Skill/模型、全局并发上限、断线续查原 session、旧 worker 不能提交 |
| P2 | 责任目录、通知 outbox、认证审批回调 | 目录决定收件人；重复回调幂等；过期/错误身份/旧版本拒绝；决定与续跑归档原子提交 |
| P3 | 跨历史分组建议、带标注的质量评测 | 分组只生成建议，不自动激活；报告区分 precision/recall/覆盖率/拒答，绑定样本和预测摘要 |
| P4 | MCP/CLI/OpenCode 接入、配置检查、集成回归 | 一个配置文件；模型不能调用专家批准；真实协议测试与真实模型/设备结果分开报告 |

## 2. 后台挖掘

显式 campaign 选择一个注册 profile 和有限 source ID 集合，冻结真实 Git before/after packet、
submission schema、Skill 内容及模型标识。后台使用 OpenCode session/prompt_async/message API。
每个 session 只返回结构化分析，禁用工具与委派；完整上下文不足时返回 needs_context，进入工作台补充调查。
后台 worker 不持有 owner、curator、代码修改或设备执行权限。

工作项保存 session ID、预分配 message ID、状态、租约代数和输出证据。
HTTP 超时不推断请求失败，不创建第二个模型请求；恢复时查询已存 session/message。
无法确定是否已发送的任务进入 attention，须操作者检查后处理。调度租约仅允许恢复查询，
不会重复发送已经进入 sending 状态的消息。结果必须再次通过原有代码引用、覆盖、反例和版本校验。
模型错误和无效输出明确归档，不无限重试。并发预算在共享数据库中统一约束。

## 3. 专家审批桥

owner → 稳定 principal + 固定收件目标由操作者配置。审批请求冻结候选版本、完整上下文摘要、
pattern 摘要、owner 目录摘要和有效期；通知 outbox 与请求同一事务建立。
SMTP 或 webhook worker 采用持久化状态和幂等消息键。网络结果不确定时保留 uncertain，
不声称 exactly-once 投递，也不自动轰炸式重发。

审批 HTTP 接口接收企业审批网关签名的结构化 JSON。签名覆盖时间戳和完整正文，
网关负责登录认证并给出稳定 principal；HMOPT 校验签名、时间窗、责任人、请求有效期、
候选/模式摘要及版本。该接口不是自行实现的 SSO，也不接受邮件自由文本作为批准。
回调、服务状态变更、认证凭证摘要、审批归档和 continuation 记录在同一事务中完成。
continuation 给出批准 ID 与候选 ID 的工作台入口；本轮默认人工触发，不能扩大原授权范围。

## 4. 质量与聚合

已分析历史可按机制文本相似度形成有限分组建议；必须经 pattern-synthesis Skill 检查跨实例差异和反例。
相似度属于检索启发式，不是语义等价证明。评测使用独立专家标签、冻结的样本摘要、数据划分与模型输出；
测试集与训练/调参集不能按相同代码家族泄漏。未知/缺失输出计入覆盖损失，不从分母静默删除。

## 5. 仍需真实部署验收的边界

本地协议测试证明恢复/门禁/认证实现，不证明模型推理准确率或真机性能收益。
真实专家目录、企业身份网关、模型服务及设备资源必须由部署方提供。
AST/CFG 跨函数证明、自动发布原生 Skill Hub 包、真机资源池属于独立后续能力，不能把当前文本扫描、
本地知识分级或 fixture 测量当作这些能力已完成。

## 6. 验证记录（2026-09-11）

- 全部 Evolution 测试加 OpenCode 命令 golden：643 passed、4 skipped，461.32 秒。
- 最后恢复逻辑、邮件简报和配置检查补充回归：41 passed。
- 兼容 stdio 配置透传、CLI schema 和最终命令更新后的接口补充回归：38 passed。
- 真实运行中的 OpenCode 接受并保存全工具 deny 的研究 session；预分配消息 ID 在 noReply 请求中往返一致。
  测试 session 只有一条 user 消息，未调用模型，随后清理。
- 临时验收工作台已识别 evolve-production（coordinator、subtask=false），统一 MCP 连接 connected。
- 真实本地 HTTP 适配器、签名网关挂载、通用 MCP 凭据不能冒充专家、原子审批、并发通知、旧 worker fencing、
  不确定远端容量保留和失败输出归档均有测试覆盖。模型回复、专家标签与 SMTP 投递使用明确的 fixture。
- Ruff 与 git diff --check 通过；所有改动及新增文本文件保持 LF。

OpenCode 消息 ID 布局依据该运行时版本的 [官方实现](https://github.com/anomalyco/opencode/blob/v1.18.30/packages/opencode/src/id/id.ts)，
接口参数同时对照本地 `/doc`。配置、启动、签名和恢复操作见 [生产使用指南](EVOLUTION_PRODUCTION_RUNBOOK_CN.md)。
