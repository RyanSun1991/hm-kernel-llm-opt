# Kernel 知识库设计方案

## 1. 背景与目标

当前仓库已经有两类与知识复用相关的机制：

- `.opencode/docs/*.md`：人工整理的 bootstrap/design/trace 文档
- `.opencode/memory/*`：面向 OpenCode pipeline 的长期记忆

这两类能力能支持单次任务复用，但还不能很好解决下面几个问题：

1. 一个知识主题可能跨多个目录、多个文件、多个符号。
2. 多个目录可能共享同一份子系统知识。
3. 知识不适合直接放在 kernel 原仓源码目录下。
4. OpenCode 在分析和优化时，还不能自动把“目标目录/目标符号”映射到对应知识库内容。
5. 现有 `bootstrap_docs` 仍然偏静态配置，`memory_files` 仍然偏路径 slug 推断，不具备目录级、子系统级、符号级的统一检索能力。

因此，这里需要设计一个独立于 kernel 原代码、但又能被 HMOPT / OpenCode 自动接入的 Kernel Knowledge Base。

## 2. 设计原则

### 2.1 核心原则

- 不侵入 kernel 原仓。
- 支持目录级、文件级、子系统级、符号级、多目录共享知识。
- 能被 OpenCode 在研究、规划、编码、review、test 各阶段自动注入。
- 既能写“稳定知识”，也能保留“局部项目覆盖层”。
- 能随着 kernel 分支和版本演进进行版本化管理。
- 检索应以确定性映射为主，以向量检索为增强，不应只依赖 embedding。

### 2.2 非目标

- 不把知识库设计成新的 wiki 系统。
- 不把所有运行时证据都塞进知识库。
- 不把一次性分析结论直接当成长期知识。
- 不要求第一版就支持复杂图数据库写回。

## 3. 现状调研结论

基于当前仓库实现，知识相关能力大致分三层：

### 3.1 现有静态知识入口

- `configs/pipeline_profiles.yaml` 中的 `bootstrap_docs`
- `.opencode/docs/*_bootstrap.md`

特点：

- 适合固定场景，如 reclaim、workqueue 等 specialist 入口
- 适合“预读文档”
- 但不能按目标路径和符号自动决策

### 3.2 现有长期记忆入口

- `.opencode/memory/targets/`
- `.opencode/memory/subsystems/`
- `.opencode/memory/global_lessons.md`

当前 `src/hmopt/opencode/pipeline.py` 的 `_infer_memory_paths()` 只会根据 target 做 slug 推断，再附加 `global_lessons.md`。这意味着：

- 只能做很粗的 target 级命中
- 无法表达“多个目录共用一份知识”
- 无法表达“某个符号需要额外知识”
- 无法表达“某个知识只对某个 branch / kernel family 生效”

### 3.3 现有代码检索入口

- `src/hmopt/indexing/llamaindex_pipeline.py`
- `src/hmopt/api/mcp_service.py`

当前 Kernel Index 已经能做：

- 向量召回
- 符号聚焦
- Neo4j graph 扩张
- rerank 和 snippet 回填

但还不能做：

- 目录/子系统/符号到知识库条目的自动映射
- 知识文档的检索和摘要注入
- 代码上下文和知识上下文的统一打包

## 4. 方案选型比较

### 4.1 方案 A：知识库放当前仓库的 `.opencode/docs/` 或 `docs/`

优点：

- 最简单，零额外仓库成本
- 与 OpenCode 工作流距离最近
- 最容易直接被 pipeline 读取

缺点：

- 当前仓库会越来越臃肿
- 不利于多个 kernel 项目共享
- 权限、版本、复用边界不清晰
- 容易把一次性分析和长期知识混在一起

适用场景：

- 单项目、单 kernel 树、团队规模小

### 4.2 方案 B：知识库放独立仓库

优点：

- 结构清晰，职责单一
- 可供多个 kernel 仓、多个优化仓共享
- 易做版本化、review、owner、发布
- 不污染 kernel 源码仓和当前 harness 仓

缺点：

- 需要额外的同步与配置
- OpenCode 首次接入要增加 resolver 逻辑
- 如果完全外置，局部项目知识覆盖不够方便

适用场景：

- 多项目共享、团队协作、长期演进

### 4.3 方案 C：独立知识库仓库 + 当前仓库本地 overlay

优点：

- 兼顾共享与项目定制
- 中央知识能跨项目复用
- 当前项目可以快速补充临时或项目专属知识
- 便于后续逐步演进为企业级知识体系

缺点：

- 结构比 A 略复杂
- 需要定义优先级和合并规则

这是推荐方案。

## 5. 推荐总体架构

推荐采用三层结构：

1. Authoritative KB Repo
2. Local Overlay KB
3. Resolver + Retrieval Integration

### 5.1 Layer 1: Authoritative KB Repo

单独建立一个知识库仓库，例如：

- `hm-kernel-kb`
- `kernel-opt-knowledge-base`

它不放在 kernel 原代码目录下，也不放在 kernel 仓内部。建议作为 sibling repo 或统一知识仓存在，例如：

```text
/work/
  hm-verif-kernel/
  hm-kernel-llm-opt/
  hm-kernel-kb/
```

或者：

```text
/knowledge/
  hm-kernel-kb/
```

### 5.2 Layer 2: Local Overlay KB

在当前仓库保留一个轻量 overlay，用于：

- 项目特有知识
- 未沉淀到中央知识仓的新增经验
- 分支特有、设备特有、版本特有说明

建议位置：

- `.opencode/kb_overlay/`

这个目录不属于 kernel 源码目录，因此符合“不放到 kernel 原代码下面”的要求。

### 5.3 Layer 3: Resolver + Retrieval Integration

在 HMOPT / OpenCode 中新增 Knowledge Resolver：

- 输入：target path、changed files、focus symbols、pipeline profile、runtime hotspots
- 输出：应加载的知识条目列表、优先级、摘要、来源

Knowledge Resolver 应在两个时机工作：

1. Pipeline staging 阶段
2. Kernel Index / MCP context retrieval 阶段

这样可以保证：

- researcher 一开始就拿到知识上下文
- coder / reviewer / tester 也能拿到针对性的知识约束
- runtime hotspot 分析可以反向补充知识映射

## 6. 知识库内容模型

知识库不应只按“文档路径”组织，应该按“知识条目”组织。

每个知识条目建议包含：

- `id`
- `title`
- `type`
- `scope`
- `content`
- `owners`
- `tags`
- `applicability`
- `relations`
- `verification`

### 6.1 条目类型

建议支持以下类型：

- `directory_overview`
- `subsystem_overview`
- `symbol_guide`
- `hotpath_note`
- `concurrency_note`
- `optimization_pattern`
- `anti_pattern`
- `validation_note`
- `cross_cutting_topic`

### 6.2 Scope 维度

知识条目应支持多种匹配维度：

- 目录前缀
- 文件路径
- 符号名 / 限定符号名
- 子系统标签
- 运行时热点标签
- 平台标签
- branch / kernel family

### 6.3 Front Matter 建议

建议每个知识条目使用 Markdown + YAML front matter：

```md
---
id: kb.memmgr.reclaim.overview
title: Memmgr Reclaim Overview
type: subsystem_overview
owners:
  - kernel-perf
repo_ids:
  - hm-verif-kernel
branches:
  - master
  - opencode
path_prefixes:
  - sysmgr/memmgr/include/reclaim/
  - sysmgr/memmgr/mem/reclaim/
symbols:
  - reclaim_services
  - wakeup_reclaimd
subsystems:
  - memmgr
  - reclaim
tags:
  - hotpath
  - instruction-count
priority: 90
shared: true
last_verified_commit: abcdef123456
review_status: reviewed
---
```

正文再写：

- 边界
- 关键入口
- 热路径
- 并发模型
- 常见优化方向
- 常见错误优化方向
- 验证要点

## 7. 推荐目录结构

### 7.1 独立知识仓结构

```text
hm-kernel-kb/
  README.md
  registry/
    repos.yaml
    path_map.yaml
    symbol_map.yaml
    subsystem_map.yaml
  entries/
    subsystems/
      memmgr/
        reclaim_overview.md
        vmpressure_note.md
      workqueue/
        scheduler_overview.md
    directories/
      sysmgr/
        memmgr/
          mem_reclaim.md
    symbols/
      reclaim_services.md
      wakeup_reclaimd.md
    cross_cutting/
      locking/
        reclaim_locking.md
      perf/
        instruction_count_patterns.md
    validation/
      build_and_test_matrix.md
```

### 7.2 当前仓库本地 overlay 结构

```text
.opencode/
  kb_overlay/
    entries/
      project_specific/
        reclaim_device_xx.md
        hyperhold_product_variant.md
    registry/
      local_path_map.yaml
      local_symbol_map.yaml
```

### 7.3 运行时 cache 结构

建议由程序维护只读 cache，不要求人工编辑：

```text
data/
  kernel_kb_cache/
    cloned_repo/
    compiled_manifest.json
    embeddings/
```

## 8. 映射模型设计

这是整个方案的关键。

### 8.1 为什么不能只靠向量检索

原因很简单：

- kernel 目录名、函数名、宏名有强结构性
- 知识命中通常是确定性的，不应该完全依赖 embedding 相似度
- 很多跨目录共享知识，本质是“规则映射”问题，不是纯语义搜索问题

因此建议：

- 第一层：规则映射
- 第二层：图和标签扩展
- 第三层：向量召回增强

### 8.2 推荐映射优先级

按以下优先级加载知识：

1. exact file match
2. deepest path prefix match
3. exact symbol match
4. subsystem match
5. cross-cutting tag match
6. global optimization / validation knowledge

### 8.3 路径映射示例

```yaml
- id: kb.memmgr.reclaim.overview
  match:
    path_prefixes:
      - sysmgr/memmgr/mem/reclaim/
      - sysmgr/memmgr/include/reclaim/
  score: 100
```

### 8.4 多目录共享示例

```yaml
- id: kb.memmgr.pressure.pipeline
  match:
    path_prefixes:
      - sysmgr/memmgr/mem/reclaim/
      - sysmgr/memmgr/psi/
      - sysmgr/memmgr/mem/stat/
  score: 85
```

### 8.5 符号映射示例

```yaml
- id: kb.symbol.reclaim_services
  match:
    symbols:
      - reclaim_services
      - reclaim_services_fastpath
  score: 95
```

### 8.6 共享知识和覆盖规则

建议优先级如下：

1. local overlay exact match
2. local overlay scoped match
3. central repo exact match
4. central repo scoped match
5. global shared knowledge

如果多个知识条目命中，不做“二选一”，而是做排序后取 Top N。

## 9. 自动获取与注入流程

### 9.1 Pipeline 阶段

在 `src/hmopt/opencode/pipeline.py` 中，现有 `_infer_memory_paths()` 之后应新增：

- `resolve_knowledge_entries()`

输入：

- `target`
- `profile`
- 可选 `hotspot_symbols`
- 可选 `changed_files`

输出：

- `knowledge_entries`
- `knowledge_summary`
- `knowledge_sources`

这些内容应写入 staged task：

- `knowledge_entries`
- `knowledge_cache_key`
- `knowledge_summary_file`

同时写入 prompt：

- `Knowledge files:`
- `Knowledge summary:`

### 9.2 Research 阶段

research agent 在正式读代码前先读取：

1. bootstrap docs
2. memory files
3. knowledge entries

这样 researcher 可以先建立子系统边界、热路径假设和历史坑点。

### 9.3 Kernel Index / MCP 阶段

在 `src/hmopt/api/mcp_service.py` 的 `kernel_index_code` 之外建议增加两种方式之一：

#### 方案 1：新增独立 MCP Tool

- `kernel_knowledge_context`

输入：

- `target`
- `symbols`
- `scenario`

输出：

- 命中的知识条目
- 摘要
- 引用文件

#### 方案 2：增强现有 `kernel_index_code`

把知识上下文作为 payload 的一部分返回：

- `knowledge_hits`
- `knowledge_summary`

推荐先做方案 1。

原因：

- 语义更清晰
- 不破坏现有 code retrieval 协议
- 便于分步上线

### 9.4 Runtime Profiling 阶段

当 flamegraph / hitrace / hiperf 识别出 hotspot symbol 后，可以二次触发：

- symbol -> knowledge mapping

这一步尤其重要，因为很多时候用户给的是目录，但真正应该读的知识是 hotspot symbol 或跨目录共享设计文档。

## 10. 配置设计建议

建议新增单独配置文件，例如：

- `configs/kernel_kb.yaml`

推荐字段：

```yaml
kernel_kb:
  enabled: true
  repo_id: hm-verif-kernel
  source:
    mode: hybrid
    central_repo_path: /work/hm-kernel-kb
    local_overlay_path: .opencode/kb_overlay
    cache_dir: data/kernel_kb_cache
  resolver:
    max_entries: 8
    enable_symbol_match: true
    enable_path_match: true
    enable_subsystem_match: true
    enable_vector_fallback: true
  retrieval:
    preload_for_pipeline: true
    expose_mcp_tool: true
    include_summary_in_prompt: true
    max_summary_chars: 4000
```

## 11. 推荐实现路径

建议分 4 期实施，而不是一次打满。

### Phase 1: 静态知识库接入

目标：

- 能从独立 repo + local overlay 读取知识条目
- 能根据 target path 自动映射 path-level knowledge
- pipeline prompt 能自动附加 knowledge files

交付：

- `configs/kernel_kb.yaml`
- `KnowledgeResolver`
- 本地/中央 manifest 解析
- pipeline 注入

### Phase 2: 符号级知识映射

目标：

- 基于 Kernel Index focus symbols 自动命中 symbol knowledge
- 在 research / hotspot_debug / patch_planning 中增强知识命中

交付：

- symbol map
- symbol-aware resolver
- `kernel_knowledge_context` MCP tool

### Phase 3: 知识摘要与向量增强

目标：

- 对命中的多个知识条目做摘要压缩
- 支持 cross-cutting topic 的相似检索

交付：

- KB entry embeddings
- summary builder
- priority + diversity rerank

### Phase 4: 知识回流闭环

目标：

- 将经过 reviewer / tester / manager 确认的稳定结论回流到知识库
- 区分一次性 run artifact 与长期有效知识

交付：

- promote-to-kb workflow
- review_status / verification metadata
- bad-plan / anti-pattern 聚合

## 12. 推荐的最终落位

这里给出明确建议。

### 12.1 不推荐

- 放到 kernel 原代码目录下
- 全部只放 `.opencode/docs/`
- 全部只靠 memory slug 命中

### 12.2 推荐落位

推荐采用：

- 中央知识仓：独立 repo
- 项目 overlay：当前仓库 `.opencode/kb_overlay/`
- 缓存和编译产物：`data/kernel_kb_cache/`
- 配置：`configs/kernel_kb.yaml`

这是目前在工程复杂度、可维护性、自动接入能力之间最平衡的方案。

## 13. 对 OpenCode / HMOPT 的具体接入建议

### 13.1 OpenCode 层

在 pipeline session 初始化时，把知识库解析结果纳入 staged task。

新增字段建议：

- `knowledge_entries`
- `knowledge_summary`
- `knowledge_sources`
- `knowledge_overlay_hits`
- `knowledge_central_hits`

### 13.2 MCP 层

新增 knowledge MCP tool，给 researcher / plan reviewer / coder / reviewer / tester 共用。

建议工具名：

- `kernel_knowledge_context`

### 13.3 HMOPT 平台层

知识库应与下面能力打通：

- kernel index
- flamegraph / hitrace / hiperf hotspot symbols
- build/test validation notes
- Docker 化交付时的附带配置挂载

也就是说，Docker 部署时建议支持：

- 挂载 central KB repo
- 挂载 local overlay
- 挂载 compiled cache

## 14. 风险与治理

### 14.1 风险

- 知识条目过期
- 多版本 kernel 语义不一致
- symbol rename 后映射失效
- 本地 overlay 和中央知识冲突
- 文档膨胀，检索噪声升高

### 14.2 治理建议

- 每个条目记录 `last_verified_commit`
- 引入 `review_status`
- 对中央知识仓设置 owner
- 对 overlay 允许更快迭代，但定期 promote / prune
- 设定 top-N 注入上限，避免 prompt 膨胀

## 15. 最终结论

如果目标是让 OpenCode 在分析和优化 kernel 代码时自动获取并映射对应知识库，最佳方案不是把知识简单塞进 `.opencode/docs/`，也不是把知识直接放进 kernel 源码树，而是：

1. 建立独立的 central kernel KB repo。
2. 在当前仓库保留 `.opencode/kb_overlay/` 作为项目覆盖层。
3. 在 HMOPT 中新增 Knowledge Resolver，按路径、符号、子系统、标签统一做确定性映射。
4. 在 pipeline 阶段和 MCP 检索阶段同时注入知识上下文。
5. 后续再把知识库接入 runtime hotspot 和长期 memory 回流。

这样做可以同时满足：

- 与 kernel 原仓解耦
- 多目录共享知识
- 多项目复用
- 自动映射
- 与 OpenCode multi-agent workflow 深度集成

## 16. 下一步建议

建议下一步直接进入实现 Phase 1，优先做下面 5 件事：

1. 新增 `configs/kernel_kb.yaml`
2. 新增 `src/hmopt/opencode/knowledge.py`
3. 实现 central repo + local overlay 的 manifest 解析
4. 在 `pipeline.py` 里接入 `resolve_knowledge_entries()`
5. 在 staged task 和 prompt 中输出 knowledge files / summary

这 5 步做完后，OpenCode 就已经能在任务启动时自动装载对应 kernel 知识库，不需要先等到完整向量化或图化方案全部完成。
