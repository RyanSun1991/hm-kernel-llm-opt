# Promotion Policy

固化设计 §8 / §9 / §4.2。决定一条候选什么时候能从 Tier1（候选）晋升到 Tier2（共享），以及 L0→L3 阶梯。

## 触发条件（满足其一 + 过三道门）

1. **≥ 2 个独立任务复现收益** — bench delta 同向，不同 target。
2. **单任务收益显著且有 bench 证据** — 附 `validation_path` + `delta_pct`。
3. **失败教训具高复用价值** — 升入 `anti_pattern`（A 系列）或 `bad_plan`（B 系列），可防止重复踩坑。

## 三道质量门（顺序通过）

```
候选 L1 ──▶ 门1 Schema/Lint/脱敏 ──▶ 门2 证据 ──▶ 门3 策展 + eval ──▶ L2 稳定
            CI 自动                   自动        Curator + 人 + eval-gate
```

**门 1 · Schema / Lint / 脱敏**（CI 自动）
- `python tools/lint.py`：每条记录过对应 JSON-Schema。
- `python tools/redact.py --check`：命中 device serial / hex key / `/dev/serial/by-id` / ssh priv key / AKID / GHP / Slack token → 拒。
- **任一失败 → block PR**。

**门 2 · 证据门**（CI 自动）
- 知识声明：`evidence[]` ≥ 1 项；引用必须能 resolve（bench 路径、commit hash、review 路径）。
- 技能编辑：必须附 `skill_patch` manifest（含 `task_suite` + `metrics`），缺一即拒。
- **无证据 → 留在 L1，不可入库**。

**门 3 · 策展 + eval**（Curator + 人 + 自动）
- **Curator-agent** 预跑：去重 / 冲突 / Pareto（详见 `merge_policy.md`）。
- **双评审签字**：1 名领域 reviewer（结论对不对）+ 1 名流程 reviewer（合规、可复用）。
- **技能**额外过 **eval-gate**：在 `eval/task_suites/<suite>/` 上 A/B；**严格变好**（pass_rate 不降、regression_rate 不增）才合入。
- **无豁免**：要破例只能「降级为 L1 候选 + owner 签字 + 下周期复核」，不得直接合入。

## L0 → L1 → L2 → L3 晋升路径

| 等级 | 判据 | 操作 |
|---|---|---|
| **L0 draft** | 仅本地、未结构化 | 在 `.opencode/local/` |
| **L1 candidate** | schema 完整 + 初始证据 | 落 `staging/<member>/<date>/*.json`，开 PR |
| **L2 stable** | 过三道门 + 双评审 | merge 到 `knowledge/` 或 `skills/domain/` 或 `skills/technique/` |
| **L3 core** | 跨 ≥ 2 个子团队复用成功 + owner 团队签字 | 升入 `skills/core/`（享更严评审 + 更高 eval 门）|

## 打分（用于晋升排序 + 衰减）

```
score = w1·evidence_strength + w2·confirmations + w3·recency
      + w4·generality          − w5·counter_evidence − w6·staleness
```

陈旧度衰减：当 `invalidation` 条件命中（如 kernel rebase）→ score 自动衰减，触发 `deprecation_policy.md`。

## 反例：什么 *不能* 直接进 hub

- 单 target / 单函数级的 fact（先归 `knowledge/targets/<slug>/facts/`，不要 hardcode 进 skill）。
- 措辞未稳定的"经验之谈"（无 evidence，留 L1 草稿）。
- 含设备 serial / key 未脱敏的原始日志（脱敏后再投）。
- 未在 `_registry/mechanisms.yaml` 注册的 mechanism（先开 PR 注册）。
