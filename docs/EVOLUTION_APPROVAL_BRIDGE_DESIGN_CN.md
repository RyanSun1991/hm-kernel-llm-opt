# Evolution 人工审批桥：实现与责任边界

状态：已实现持久化请求、责任目录、SMTP TLS/webhook outbox、认证网关回调和审批归档；企业网关与真实通道需部署接入。
当前代码和配置以 [生产使用指南](EVOLUTION_PRODUCTION_RUNBOOK_CN.md) 为准。

## 从责任人到批准归档

1. Discovery owners 按路径匹配稳定 owner，production.approval.contacts 将其映射为 principal 和通知目标。
2. evolution_request_approval 冻结候选版本、完整上下文、pattern、目录摘要和有效期，原子创建审批请求与通知。
3. 操作者运行 notify 或 notify --watch，按目录投递；Agent 不指定收件人，不替专家确认。
4. 企业网关负责登录和专家页面，对 context/decide 请求签名；统一 API 校验签名、身份、版本、证据与原门禁。
5. 确认后原子保存不可变 approval_id、认证凭证摘要、candidate 状态和 continuation。
6. OpenCode 从 dossier/queue 读取批准 ID 与 candidate ID，按用户明确选择的范围进入既有执行流程。

通知只表示请求评审，确认只表示允许进入方案阶段。方案/代码独立评审与验证门没有被替代。

## 身份保证

本地 decide/apply-sheet 仍属于可信操作者标签模式。网关回调增加 HMAC-SHA256 认证，签名覆盖路径、时间戳、正文。
principal 必须来自企业网关验证过的登录身份，且同时匹配请求冻结的目录与当前配置。
这不是 HMOPT 自建 SSO。工作台 actor 标签不能替代企业身份，网关密钥不能交给浏览器或 Agent。

## 一致性和失败处理

- 重复回调返回同一结果，改变决定或旧版本均冲突。
- 审批状态、服务门禁、receipt、continuation 共享 SQLite 事务，失败不留下部分批准。
- 目录、候选或 pattern 改变，需要重新审阅；旧决定不自动迁移到新版本。
- 通知 uncertain/sending 不自动重发；操作者检查投递后明确重试，并接受可能重复送达。
- 邮件自由文本、已读回执和链接预览不触发批准。
- continuation 默认 automatic_execution=false；原批准不扩大到全流程自动执行。

## 代码和协议

- src/hmopt/evolution/production.py：严格配置、目录与环境变量引用。
- src/hmopt/evolution/approval.py：冻结请求、outbox、认证、决定和归档。
- src/hmopt/api/evolution_approval.py：统一 HTTP 服务下的 context/decide 接口。
- src/hmopt/evolution/catalog.py：dossier 中的 approval_request、receipt 和 continuation。
- tests/test_evolution_production.py：错误身份、签名、过期、并发投递与原子回滚。

具体配置、签名正文、HTTP 错误码、工作台命令及恢复步骤见 [生产使用指南](EVOLUTION_PRODUCTION_RUNBOOK_CN.md)。
本轮没有向真实专家发送通知；本地 fixture 验证不代表企业 SSO、SMTP 或生产设备已经验收。
