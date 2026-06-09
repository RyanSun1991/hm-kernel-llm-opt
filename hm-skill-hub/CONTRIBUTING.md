# 贡献协议

## 沉淀流程

```
你本地 pipeline 跑完
  → hmopt sediment 自动产出 Tier1 候选（→ .opencode/local/sediment_staging/）
  → hmopt sediment --bundle --open-pr 打包成 PR 到本仓 staging/
  → CI: lint + secret-scan + dedup
  → Curator-agent 标注合并方案（去重/冲突/Pareto）
  → 双评审（1 领域 reviewer + 1 流程 reviewer）签字
  → merge 到 knowledge/ 或 skills/
```

详见 `policies/promotion_policy.md`。

## 三类记录的最小要求

| 类型 | 位置 | ID 前缀 | 必填要点 |
|---|---|---|---|
| bad_plan | `knowledge/global/bad_plans/` 或 `knowledge/subsystems/<sub>/bad_plans/` | `B` | mechanism（用 `_registry/mechanisms.yaml` 里的名字）+ target_pattern + scope + applies_to + reason + ≥1 evidence |
| global_lesson | `knowledge/global/{heuristics,anti_patterns,validation_pitfalls}/` | `H` / `A` / `V` | lesson + applies_when + do_or_dont + tags + ≥1 evidence + confidence |
| memory_item (target/subsystem 事实) | `knowledge/targets/<slug>/facts/` 或 `knowledge/subsystems/<sub>/` | `F` / `G` / `A` / `R` | type + title + scope + ≥1 source |
| idea_ledger 行 | `knowledge/targets/<slug>/idea_ledger.md` | `L` | mechanism + scope + status + verdicted_by/at + rationale；landed 还需 delta_pct/validation_path |
| skill | `skills/{core,technique,domain}/<name>/SKILL.md` | — | YAML frontmatter 过 `schemas/skill_frontmatter.schema.json`；变更必须附 skill_patch（带 eval metrics） |

**stable ID**：每个域内单调递增三位数字，**永不复用、永不删除**（status 改为 `superseded` 或 `deprecated`）。

## 写前自检

```bash
pip install -r tools/requirements.txt
python tools/parse_memory.py knowledge/global/anti_patterns/A001-*.md   # 看结构是否解出来
python tools/lint.py                                                    # 全量 schema 校验
python tools/redact.py --check                                          # 密钥扫描
```

## 机制（mechanism）命名

- **新机制必须先入 `_registry/mechanisms.yaml`**，CI 拒未注册的 mechanism。
- 命名规则：`kebab-case`，动词+对象（`hoist-invariant`、`batch-coalesce`）。
- 旧叫法写到 `aliases:` 数组里，参与 fuzzy match。

## 评审

- 任何 PR 需 **1 名领域 reviewer**（看结论对不对）+ **1 名流程 reviewer**（看格式/可复用性）。
- `skills/core/` 变更走更严评审（owner 团队批准）。
- **无豁免门**：eval 不达标只能降级为 L1 候选 + owner 签字 + 下周期复核。
