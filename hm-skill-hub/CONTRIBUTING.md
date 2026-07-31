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

## 落盘格式：一记录一文件 + frontmatter（Phase 0.5 收敛，设计 §6.1 / §7）

每条 knowledge 记录 = **单独一个 `.md` 文件** = YAML frontmatter（全 schema 字段）+ markdown body：

```markdown
---
id: F001
type: pattern
title: ...
scope: {level: function, subsystem: mm-reclaim, target_slug: mm-vmscan-shrink_node}
source: [{kind: bench, ref: ...}]
maturity: L2
status: active
created_at: 2026-04-26T10:00:00Z
---

# 人类可读标题

free-form markdown 正文（memory_item 的 `body` 字段从这里取）。
```

**路径即 scope（CI 强校验）**：文件路径编码 scope，与 frontmatter 必须一致，不一致 `lint.py` 精准报错。
- 旧的「`### A001` 标题 + `- **key**: value` 多记录堆叠」格式已废弃，不再支持。
- 禁止每类自扩 markdown 字段；frontmatter 字段一律走对应 schema。

## 记录类型的最小要求

| 类型 | 位置（路径编码 scope）| ID 前缀 | 必填要点 |
|---|---|---|---|
| bad_plan | `knowledge/global/bad_plans/<B###>.md`（全局 `applies_to.subsystems: ['*']`）或 `knowledge/subsystems/<sub>/bad_plans/<B###>.md` | `B` | mechanism（用 `_registry/mechanisms.yaml` 里的名字）+ target_pattern + scope + applies_to + reason + ≥1 evidence |
| global_lesson | `knowledge/global/{heuristics,anti_patterns,validation_pitfalls}/<ID>.md` | `H` / `A` / `V` | lesson + applies_when + do_or_dont + tags + ≥1 evidence + confidence（路径叶目录须与 kind 一致）|
| memory_item (target/subsystem 事实) | `knowledge/targets/<slug>/facts/<F###>.md` 或 `knowledge/subsystems/<sub>/<ID>.md` | `F` / `G` / `R` | type + title + body + scope + ≥1 source + maturity + status + created_at（`scope.target_slug`/`scope.subsystem` 须与路径一致）|
| idea_ledger 条目 | `knowledge/targets/<slug>/idea_ledger/<L###>.md`（**每条一文件**）| `L` | mechanism + target_slug + scope + status + verdicted_by/at + rationale；landed 还需 delta_pct/compare_level/validation_path |
| skill | `skills/{core,technique,domain}/<name>/SKILL.md` | — | YAML frontmatter 过 `schemas/skill_frontmatter.schema.json`；变更必须附 skill_patch（带 eval metrics） |

**stable ID**：每个域内单调递增三位数字，**永不复用、永不删除**（status 改为 `superseded` 或 `deprecated`）。
**关系边**：时态/矛盾用 `supersedes[]/superseded_by[]`；泛化包含用 `subsumes[]/subsumed_by[]`（具体实例永远保留为证据，不被吞）。

## 证据层级（策展强度判定）

从强到弱统一排序：

1. 客观测试或基准结果；
2. 实际落地或回退结果；
3. 人的明确裁决；
4. 独立成员/任务复用；
5. 可定位的静态代码事实；
6. 工具输出；
7. 模型自评。

模型的 `confidence` 只是元数据，**不是证据**。候选可以带弱证据进入 L1，但不得仅凭模型自信
晋升、覆盖或 supersede 既有知识。发生冲突时，Curator 必须按上述层级比较新旧证据；无法证明
新证据严格更强时，保留两者并交给人工复审。

## 写前自检

```bash
pip install -r tools/requirements.txt
python tools/parse_memory.py knowledge/targets/mm-vmscan-shrink_node/facts/F001-*.md  # 看 frontmatter 是否解成 schema object
python tools/lint.py                                                    # 全量 schema 校验 + 路径/scope 一致性
python tools/redact.py --check                                          # 密钥扫描
python tools/tests/test_tools.py                                        # 工具链自测（无需 pytest）
```

## 机制（mechanism）命名

- **新机制必须先入 `_registry/mechanisms.yaml`**，CI 拒未注册的 mechanism。
- 命名规则：`kebab-case`，动词+对象（`hoist-invariant`、`batch-coalesce`）。
- 旧叫法写到 `aliases:` 数组里，参与 fuzzy match。

## 评审

- 任何 PR 需 **1 名领域 reviewer**（看结论对不对）+ **1 名流程 reviewer**（看格式/可复用性）。
- `skills/core/` 变更走更严评审（owner 团队批准）。
- **无豁免门**：eval 不达标只能降级为 L1 候选 + owner 签字 + 下周期复核。
