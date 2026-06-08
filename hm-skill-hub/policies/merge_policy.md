# Merge Policy

固化设计 §10。**两类资产，两台合并引擎**，不可混用。

## 引擎 A — Knowledge（追加型）：集合并 + 去重 + 冲突消解

**绝不用 git 行级合并。** Curator-agent 在 PR 上跑：

```
for item in incoming:
    dup = near_duplicate(item, hub_items)       # embedding 相似度 ≥ 0.92
    if dup:
        merge_provenance(dup, item)              # 合并 source[]；confirmations += 1
        continue
    conflict = contradiction(item, hub_items)    # 同 (target, mechanism) 断言相反
    if conflict:
        if stronger_evidence(item):              # 证据/新近度加权（Zep 双时态）
            conflict.status = "superseded"
            conflict.valid_until = now()
            item.supersedes = [conflict.id]
            add(item)
        elif high_risk:
            escalate_to_human(item, conflict)
        else:
            drop_with_citation(item, conflict)
    else:
        add(item)
```

**CRDT 纪律**：追加 + tombstone（`active / superseded / deprecated`），**永不删除**。

## 引擎 B — Skills（编辑型）：SkillOpt 验证门 + GEPA Pareto

**绝不用集合并。** 每个 skill 改动 = 一份 `skill_patch` manifest（有界 add/del/replace）：

```
def merge_skill_edit(skill, edit):
    if edit ∈ skill.bad_edits:        return REJECT("known-bad edit")
    edit = clip_to_budget(edit, textual_learning_rate)
    cand = apply(skill, edit)
    score = run_evals(cand, skill.eval_suite)
    if score.strictly_better_than(skill.score):
        skill = cand
        write_scorecard(skill, score)
    else:
        skill.bad_edits.append(edit)
    pareto = update_pareto(pareto, cand, per_instance_scores)   # 互补候选→ candidates/
```

- **文本学习率** = 每次发布的有界编辑预算。
- **Pareto 前沿** = 多人提编辑时保留「各自在某实例最优」的一组候选到 `skills/<name>/candidates/`，定期合并互补 lesson。

## 双评审

每个 PR 需：
- **1 名领域 reviewer**（结论对不对、机制合理性）；
- **1 名流程 reviewer**（schema 合规、stable ID 唯一性、双时态字段完整、未被去重源命中）。

`skills/core/` 改动 = **2 名 owner** 必须签字。

## 无豁免

`metrics.pass_rate` 不增、`regression_rate` 增加 → 一律拒。**唯一破例路径**：降级为 L1 候选 + owner 签字 + 下个评测周期复核。**不允许 "merge despite eval"**。

## 脱敏（贯穿所有合并）

`tools/redact.py` 命中以下即拒：

| pattern | 例 |
|---|---|
| `aws-akid` | `AKIA...` |
| `ssh-priv` | `-----BEGIN ... PRIVATE KEY-----` |
| `generic-hex-key` | 长度 ≥ 40 的 hex 串 |
| `device-serial` | `serial=...` / `imei=...` |
| `dev-serial-path` | `/dev/ttyUSB<N>` / `/dev/serial/by-id/<…>` |
| `github-pat` | `ghp_...` |
| `slack-token` | `xox?-...` |

人工脱敏后用 `[REDACTED]` 占位即可重提。
