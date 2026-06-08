# Deprecation Policy

固化设计 §13.3。失效治理：什么时候记录从「在用」走向「已被取代 / 已废弃」，以及怎样**保留可审计**而不污染检索。

## 状态机

```
   active ──新证据更优──▶ superseded   （旧条目 valid_until=now，新条目 supersedes=[旧id]）
      │
      │ invalidation 命中
      │   OR   score 长期衰减
      │   OR   连续 N 次反例
      ▼
   deprecated   （从检索 / inline 排除；不物理删除）
```

**状态字段是 schema 强制的**（`status: active | superseded | deprecated`）。**永不物理删除**——这是可审计与"为什么以前这么做"调查的底线。

## 触发条件

| 触发 | 动作 |
|---|---|
| 同 (target, mechanism) 出现更强证据 | Curator 自动把旧条目设 `superseded`、`valid_until=now()`、新条目 `supersedes=[old.id]` |
| `invalidation` 字段命中（例：`"rebase 后须重校 offset"`）| 标 `deprecated`；附 deprecation_reason |
| 连续 ≥ 3 次反例 evidence | 标 `deprecated`；写一条 anti_pattern A### 解释为什么之前以为它对 |
| 提到的 mechanism 从 `_registry/mechanisms.yaml` 移除 | 该记录 `status=deprecated`，留 reference 给 successor mechanism |
| skill 的 eval pass_rate 连续 ≥ 2 个周期下降 | skill `status=deprecated`；保留 `best_skill.md` 历史快照；切到 Pareto 前沿里的另一个候选 |

## 定期清理（Phase 4 nightly 作业）

```
nightly:
  scan all records with status ∈ {superseded, deprecated}
  → keep in repo (audit trail)
  → exclude from RAG index rebuild
  → exclude from @-inline context assembly
  → tag with deprecation_reason if missing
```

## 重激活（revisit）

`deprecated` 不是终点。若新证据显示之前的判断错了：
1. 不改旧条目 status（保留历史）。
2. 写一条**新**记录（新 ID），在 `related_ids` 引用旧条目。
3. 注释 `"revisits B017: new evidence in bench/<...>"`。

idea_ledger 的 `deferred + reopen_trigger` 机制是同一思路的另一形态——deferred 不是 deprecated，是「条件成熟可重提」。

## 不做的事

- ❌ 物理删除任何记录（除非违法/泄密；用 `git filter-repo` 必须经 owners 团队批准 + 留 ADR）。
- ❌ 改 stable ID。
- ❌ 直接 reset `confirmations / score`；应当通过新增 counter-evidence 让 score 自然衰减。
