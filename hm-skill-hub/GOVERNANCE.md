# 治理

## 角色

| 角色 | 职责 |
|---|---|
| **Owners (`skills/core/`)** | 核心技能的最终守门人；批准 L2→L3 晋升；review 任何 core skill 变更 |
| **Domain reviewer** | 内容评审（结论对不对、证据强不强、是否泛化）|
| **Process reviewer** | 流程评审（schema 合规、是否被去重源命中、双时态字段是否完整）|
| **Curator-agent** | CI 自动跑：去重 / 冲突检测 / Pareto / 给出合并方案 |
| **Maintainers** | 仓库基础设施、CI、发布工具、`_registry/` 维护 |

## CODEOWNERS 约定

待拆仓后落到根 `.github/CODEOWNERS`。Phase 0 先用如下规则作为参考：

```
/skills/core/                 @core-skill-owners
/_registry/                   @maintainers
/policies/                    @maintainers @core-skill-owners
/schemas/                     @maintainers
/knowledge/global/            @global-knowledge-curators
/knowledge/subsystems/mm-*/   @mm-team
/knowledge/subsystems/wq-*/   @wq-team
```

## 分支保护（拆仓后建议）

- `main`：禁止直 push；PR 必须 ≥ 1 owner + ≥ 1 process reviewer；CI 全过；要求线性历史。
- `skills/core/**` 改动需 ≥ 2 owner 批准。

## 发布节奏

| 频率 | 形式 | 触发 |
|---|---|---|
| 每周 | patch / minor 小版本 | nightly 优化作业（Phase 4）通过 + 累积变更 |
| 每月 | minor / major 稳定版 | 含 `skills/core/` 改动或 schema 演进 |
| 临时 | hotfix | 仅当 eval-gate 检出生产回归 |

每次发布产物：semver tag、`releases/skill-memory-<semver>/` 快照、release notes、自动开 PR 到业务仓更新 `skill-memory.lock`。

## 冲突 / 异议

- 内容冲突（同 target/mechanism 断言相反）：Curator 用 Zep 双时态机制自动消解（新证据胜，旧条目标 `superseded` 保留）。高风险（如涉及 `skills/core/`）升级到 owner 会议。
- 流程异议（评审拒绝但贡献者不服）：留 PR 评论 24h；未达共识则 maintainers 仲裁。

## 安全

- 任何 PR 必过 `tools/redact.py --check`；命中 secret pattern 即拒。
- 设备序列号、签名 key 一律 `[REDACTED]`；详见 `policies/merge_policy.md` 脱敏段。
