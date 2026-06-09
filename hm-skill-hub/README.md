# hm-skill-hub

团队级 skill / memory 资产仓。配套设计文档：`docs/Team_Skill_Hub_Design_CN.md`（v2.1）。

> **现状**：作为业务仓的子目录孵化中（`hm-skill-hub/`）。Phase 0–1 验证完后用
> `git subtree split --prefix=hm-skill-hub` 拆出独立仓，业务仓改 submodule pin。

## 目录速览

```
schemas/        7 份 JSON-Schema：bad_plan / global_lesson / memory_item / idea /
                skill_frontmatter / skill_patch / scorecard
_registry/      mechanisms.yaml（控制词表）+ subsystem_selectors.yaml（子系统→拓扑）
policies/       promotion / merge / deprecation 三份治理策略
skills/         core/  technique/  domain/   ——按"种类"分层；不按拓扑分
knowledge/      global/{heuristics,anti_patterns,validation_pitfalls,bad_plans}/
                subsystems/<sub>/  targets/<slug>/  index/
evidence/       benchmarks/  regressions/
eval/           task_suites/  scorecards/
staging/        Tier 1 入站候选（PR 前落区）
releases/       发布快照
tools/          parse_memory.py · lint.py · redact.py · requirements.txt
.github/workflows/ci.yml   （拆仓后激活）
```

## 快速校验

```bash
pip install -r tools/requirements.txt
python tools/lint.py            # 全量 schema lint
python tools/redact.py --check  # 密钥扫描
```

## 加一条记录

1. 看 `policies/promotion_policy.md` 确认晋升触发条件已满足；
2. 在对应目录复制示例文件，分配下一个稳定 ID；
3. 本地跑 `python tools/lint.py`；
4. 提 PR，触发 CI + 双评审（见 `policies/merge_policy.md`）。
