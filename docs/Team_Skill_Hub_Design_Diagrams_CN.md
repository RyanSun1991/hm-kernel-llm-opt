# Team Skill Hub 方案图示（配套可视化）

| 项 | 值 |
|---|---|
| 文档状态 | Draft v2（配套 `Team_Skill_Hub_Design_CN.md` v2.2）|
| 日期 | 2026-06-09 |
| 说明 | 7 张 Mermaid 图，GitHub / VS Code 直接渲染。先看图建立整体直觉，细节回主文档对应 § |
| 修订 | v2：新增图 7「读路径：混合检索 + 上下文预算」（配套主文档 v2.2 §12 改写） |

整体顺序：① 闭环全景 → ② 两套引擎（脊柱）→ ③ 沉淀漏斗 → ④ skills 维度归宿 → ⑤ 运行时组合 → ⑥ 路线图 → ⑦ 读路径（v2 新增）。

---

## 图 1 · 闭环全景（整体）—— 对应主文档 §5

两个仓库 + 四步闭环：消费 → 蒸馏 → 沉淀合并 → eval 发布 → 再消费。

```mermaid
flowchart LR
  subgraph HUB["hm-skill-hub · Tier2 团队共享 · semver"]
    direction TB
    SK["skills/<br/>引擎B：SkillOpt + Pareto"]
    KN["knowledge/<br/>引擎A：memU 合并"]
  end
  CUR{{"Curator + CI<br/>去重 · 冲突消解 · eval-gate · 脱敏 · Pareto"}}
  subgraph BIZ["业务仓 .opencode/ · 执行面 · 每成员"]
    direction TB
    PIPE["pipeline 运行<br/>research → plan → code → review → test → decision"]
    LOC["local/<br/>运行证据 + 在途记忆"]
    STG["sediment_staging<br/>Tier1 候选"]
  end
  HUB -.->|"① 消费：submodule pin + lock + 检索"| PIPE
  PIPE -->|"② 蒸馏 Tier0→1"| LOC
  LOC --> STG
  STG -->|"③ 沉淀 PR"| CUR
  CUR -->|"④ 发布：eval-gate 通过 → 升版本 + tag"| HUB
```

---

## 图 2 · 两类资产 / 两套引擎（脊柱·具体）—— 对应主文档 §4.1 / §10

一条候选按「是程序还是事实」分流，走两套完全不同的合并引擎。

```mermaid
flowchart TB
  IN["团队成员的本地沉淀候选"] --> Q{"是『程序』<br/>还是『事实』?"}
  Q -->|"过程性 Skills"| B1
  Q -->|"事实 Knowledge"| A1
  subgraph EB["引擎 B · Skills（就地编辑·绝不集合并）"]
    direction TB
    B1["有界编辑 add/del/replace<br/>文本学习率裁剪"] --> B2{"留出 eval<br/>严格变好?"}
    B2 -->|"是"| B3["更新 best_skill.md + scorecard"]
    B2 -->|"否"| B4["进 bad_edits 缓冲<br/>（不再重试）"]
    B3 --> B5["GEPA Pareto<br/>保留互补候选，防塌缩"]
  end
  subgraph EA["引擎 A · Knowledge（追加合并·绝不行级合并）"]
    direction TB
    A1["稳定 ID · 集合并"] --> A2{"近似重复?"}
    A2 -->|"是"| A3["合并出处<br/>confirmations + 1"]
    A2 -->|"否"| A4{"矛盾?"}
    A4 -->|"是"| A5["证据/新近度加权<br/>旧记录标 superseded（不删）"]
    A4 -->|"否"| A6["新增"]
  end
```

---

## 图 3 · 沉淀漏斗：三层 + 三道门 + L0–L3（具体）—— 对应主文档 §4.2 / §8 / §9

一条原始轨迹如何层层过门、晋升为团队资产。

```mermaid
flowchart TB
  T0["Tier0 运行轨迹<br/>plans · reviews · bench · design（本地，可能含密钥）"]
  T0 -->|"收口点蒸馏 hmopt sediment"| T1["Tier1 候选 = L1<br/>类型化 + 出处 + 证据（staging）"]
  T1 --> G1{"门1 Schema / Lint / 脱敏"}
  G1 -->|"拒"| X1["打回 / 留本地"]
  G1 -->|"过"| G2{"门2 证据门<br/>引用 · delta · ≥N 确认"}
  G2 -->|"无证据"| X2["留 L1"]
  G2 -->|"过"| G3{"门3 策展 + eval<br/>Curator + 双评审 + eval-gate"}
  G3 -->|"拒 / 破例"| X3["降级 L1 + owner 签字 + 复核<br/>（无豁免口子）"]
  G3 -->|"过"| T2["Tier2 = L2 stable<br/>knowledge/ 或 skills/domain/"]
  T2 -->|"跨子团队复用成功"| L3["L3 core<br/>skills/core/（组织金标准）"]
  classDef gate fill:#fff3cd,stroke:#d39e00
  classDef asset fill:#d4edda,stroke:#28a745
  class G1,G2,G3 gate
  class T2,L3 asset
```

---

## 图 4 · skills/ 维度归宿：消灭组合爆炸（具体）—— 对应主文档 §6.2

kernel 多维度各有唯一归宿；**只有 3 个维度是真正的 skill 文件夹**，file/function 全部沉到 `knowledge/`。

```mermaid
flowchart LR
  subgraph DIM["kernel 各维度"]
    direction TB
    D1["流程 / 跨切面"]
    D2["优化招式 mechanism"]
    D3["子系统 subsystem"]
    D4["目录 dir"]
    D5["文件 file"]
    D6["函数 function / symbol"]
  end
  D1 --> C["skills/core/"]
  D2 --> T["skills/technique/"]
  D3 --> M["skills/domain/ 按子系统/"]
  D4 -.->|"applies_to.path_globs<br/>（不单独建目录）"| M
  D5 --> K1["knowledge/targets/facts/"]
  D6 --> K2["knowledge/ idea_ledger<br/>+ symbol_selectors"]
  classDef skill fill:#cfe2ff,stroke:#0d6efd
  classDef know fill:#ffe5d0,stroke:#fd7e14
  class C,T,M skill
  class K1,K2 know
```

---

## 图 5 · 运行时组合：组合而非枚举（具体）—— 对应主文档 §6.2 / §12

给定一个 target，`resolver` 按 selector 解析、组合小技能、挂载知识——`(子系统 × 招式)` 矩阵在加载期消化。

```mermaid
flowchart TB
  TGT["target（例：mm/vmscan.c::shrink_node）"] --> R["resolver.py<br/>对照 clangd/scip 索引解析 selector"]
  R -->|"selector 匹配"| DOM["domain/mm-reclaim skill"]
  DOM -->|"requires"| CORE["core/*（funnel · instr-count · stage-gate）"]
  DOM -->|"requires"| TECH["technique/*（hoist · batch · lock-split）"]
  R -->|"检索挂载"| KN["该 target 的 knowledge<br/>facts + idea_ledger"]
  CORE --> CTX["组合进 agent 上下文<br/>（加载期组合，非树里枚举）"]
  TECH --> CTX
  DOM --> CTX
  KN --> CTX
```

---

## 图 6 · 分阶段路线图（整体）—— 对应主文档 §14

Phase 3（建 eval 套件）是关键长杆，红色标注。

```mermaid
flowchart LR
  P0["Phase0 抽取 · 1-2w<br/>双仓 pin + 路径兼容"] --> P1["Phase1 蒸馏 · 2-3w<br/>hmopt sediment"]
  P1 --> P2["Phase2 策展+合并 · 3-6w<br/>引擎A + CI + policies"]
  P2 --> P3["Phase3 eval门 · 6-10w ★<br/>core suite + 引擎B"]
  P3 --> P4["Phase4 自动优化 · 10w+<br/>定时作业 + 发布节奏"]
  classDef hard fill:#f8d7da,stroke:#dc3545
  class P3 hard
```

---

---

## 图 7 · 读路径：混合检索 + 上下文预算（v2 新增）—— 对应主文档 §12

resolver 在 pipeline 各阶段并行查 hub + local；scalar 过滤先于 BM25 + 向量融合，最后按 stage 预算裁切注入。

```mermaid
flowchart TB
  STG["pipeline stage<br/>(research / plan / code / review / test)"]
  TGT["target slug + symbol<br/>(mm/vmscan.c::shrink_node)"]
  STG --> R["resolver.py"]
  TGT --> R
  R -->|"selector 匹配"| SK["hub.skills<br/>domain → requires(core+technique)"]
  R -->|"target-anchored query"| RH["retrieve(hub.knowledge)"]
  R -->|"target-anchored query"| RL["retrieve(local.memory)"]
  subgraph HYB["混合检索栈（hub / local 各一份）"]
    direction TB
    SF["① scalar 预过滤<br/>status=active · maturity∈{L2,L3}<br/>scope.subsystem · target_slug"]
    BV["② BM25 + 向量 cosine + 实体加成<br/>RRF 融合"]
    SC["③ score 字段加权<br/>(§9 晋升打分喂回排序)"]
    SF --> BV --> SC
  end
  RH --> HYB
  RL --> HYB
  HYB --> MRG["合并去重<br/>同稳定 ID 以 hub 为准<br/>local 仅补未晋升的"]
  SK --> CTX["按 stage 预算裁切<br/>research:8/3K · plan:5/2K · code:3/1.5K · review:5/2K"]
  MRG --> CTX
  CTX --> AG["注入 agent context"]
  CTX -->|"落盘 retrieval.jsonl<br/>(可观测·喂衰减/deprecation 候选)"| OBS[("retrieval log")]
  classDef hub fill:#cfe2ff,stroke:#0d6efd
  classDef local fill:#ffe5d0,stroke:#fd7e14
  classDef gate fill:#fff3cd,stroke:#d39e00
  class RH,SK hub
  class RL local
  class SF,BV,SC,CTX gate
```

> **关键不变式**：markdown 是真相源，`index/` 是派生缓存——cascade 增量重嵌即可，可任何时刻整树重建；切换 faiss / pgvector / LanceDB 不影响数据。

---

**看完图后回主文档**：脊柱与原则 §4 · 布局 §6 · 数据模型 §7 · 合并引擎伪代码 §10 · 检索与运行时组合 §12 · 治理稳定可用 §13 · 风险 §15。
