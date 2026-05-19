# scip-clang 集成测试验证步骤（Step-by-Step）

> 配套：`docs/scip_clang_integration_plan.md`（设计计划）
> 适用阶段：Phase 0–5 已合入主线（截至最新 commit）
> 目标：从 zero state 推到"scip-clang 与 clangd backend 在真实内核子树上结果可比对"

## 0. 验证分层总览

按"投入成本递增 / 信号强度递增"分 7 层。**每层都是上一层的超集**：如果只关心模块没崩，跑到 L1 即可；如果要对外宣称 backend 上线，必须跑到 L7。

| 层 | 名称 | 需要 | 时长 | 出口信号 |
|---|---|---|---|---|
| L0 | 工具链清单 | — | 1 min | 该装的都装了 |
| L1 | protobuf 与 scip_pb2 自检 | python + protobuf | 30 s | `scip_pb2.Index` 可构造 |
| L2 | 翻译模块单测 | + pytest | 1 s | `test_scip_translation` 全过 |
| L3 | backend 合成 `.scip` 单测 | 同 L2 | 1 s | `test_scip_clang_backend` 全过 |
| L4 | config / CLI / 调用链 plumbing 单测 | + pydantic + pyyaml | 1 s | `test_config_indexing_backend` + `test_call_chain_call_site` 全过 |
| L5 | 玩具 C 项目端到端 | + scip-clang 二进制 + clang | 1–2 min | 真 `.scip` 文件 → Neo4j 写入成功 |
| L6 | 真实内核子树端到端 | + 一个 kernel 子目录 + compile_commands.json | 5–60 min | `hmopt index-kernel --backend scip-clang` 完成且边数与 clangd 同阶 |
| L7 | clangd ↔ scip-clang A/B 等价校验 | 同 L6 + clangd | 1–2 h | A/B 报告 `(src_id, dst_id, kind)` 重合度 ≥ 80% · call-site 覆盖 ≥ 95% |

> 沙盒里跑通的"基准线"：L0–L4。本仓库当前 commit 已通过这 45 个离线单测。

---

## L0 — 工具链清单

```bash
# 必备
python3 --version                # ≥ 3.10
pip show protobuf | grep Version # ≥ 4.25
python3 -c "import pydantic, yaml, pytest"  # 一句话校验四件套

# Phase 3+ 需要
which scip-clang                 # 应有；否则 https://github.com/sourcegraph/scip-clang/releases
which clang                      # 需与 scip-clang 同一主版本（≥ 13）

# A/B 时需要
which clangd                     # 与 clang 同一主版本

# 重新生成 scip_pb2（可选；已 check-in，一般不需要）
which protoc                     # 或 pip install grpcio-tools
```

**检查点**：

- `pip install -e ".[dev]"` 在干净 venv 里不报错（如未编辑安装，所有 `hmopt` 命令将走 `python -m hmopt.cli`）
- `bash scripts/gen_scip_pb2.sh --check` 报告 protoc 状态而不崩

**如果只有部分工具**：
- 缺 `scip-clang` → 可跑到 **L4** 为止，离线层完整覆盖了翻译/解析/配置/查询代码路径
- 缺 `clangd` → 可跑到 **L6**，只是 A/B 跳过

---

## L1 — protobuf 与 scip_pb2 自检

最便宜的"接电"测试，确认 protobuf 运行时与 check-in 的 `scip_pb2.py` 版本兼容。

```bash
python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location(
    'scip_pb2', 'src/hmopt/indexing/_generated/scip_pb2.py'
)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('Index ok:', hasattr(m, 'Index'))
print('Definition role:', m.Definition)
print('SymbolInformation Kind values sample:', m.SymbolInformation.Kind.values()[:5])

# 最小往返：构 Index → 序列化 → 反序列化
idx = m.Index(); idx.metadata.project_root = 'file:///tmp/x'
data = idx.SerializeToString()
idx2 = m.Index(); idx2.ParseFromString(data)
assert idx2.metadata.project_root == 'file:///tmp/x'
print('Index round-trip ok')
"
```

**期望输出**（实测）：

```
Index ok: True
Definition role: 1
SymbolInformation Kind values sample: [0, 66, 72, 1, 2]
Index round-trip ok
```

**失败模式**：
- `ImportError: cannot import name 'runtime_version'` → 升级 `protobuf>=4.25`：`pip install -U "protobuf>=4.25"`
- `TypeError: Couldn't build proto file ... DescriptorPool` → 通常是新旧 protoc 生成代码混用；删 `_generated/scip_pb2.py` 并重跑 `bash scripts/gen_scip_pb2.sh`

---

## L2 — 翻译模块单测（pure python）

`_scip_translation.py` 是把 SCIP 符号描述符翻译成项目内规范 ID（`path:qualname:line:kind`）的纯函数模块，零外部依赖。

```bash
python3 -m pytest tests/test_scip_translation.py -v 2>&1 | tail -40
```

**期望**：`27 passed`。覆盖：

- C 函数 / struct / struct field / macro 符号解析
- C++ namespace、method、disambiguator
- 反引号包裹的特殊文件名作为 namespace
- local 符号（`local 0`）
- 空 / 畸形符号 → 返回 None
- range 归一化（3 ints vs 4 ints）
- 范围包含与最内层 enclosing definition
- `syntax_kind` → call/type/macro 区分
- `SymbolKind` → 项目 kind 字符串映射

**失败模式**：
- 任意一个 `test_parse_*` 挂 → 翻译规则被改坏了，回到 `_scip_translation.py` 检查改动
- 整段 skip → pytest 没识别到测试函数，检查 pytest 版本

---

## L3 — backend 合成 `.scip` 单测（核心闭环）

不需要真的 scip-clang 二进制。测试在内存里构 protobuf `Index`，序列化到临时文件，喂给 `ScipClangBackend._parse_scip`，验证：

- 定义 occurrence → `CodeChunk`（含 `backend_origin="scip-clang"`、`scip_symbol`、行号转 1-based）
- 引用 occurrence → `CodeRelation`（含 `call_site_path/line/col`、`syntax_kind`）
- 跨 TU 的外部引用 → `dst_id` 以 `external::` 开头
- local 符号 / punctuation 不入图

```bash
python3 -m pytest tests/test_scip_clang_backend.py -v 2>&1 | tail -20
```

**期望**：`3 passed`。

**失败模式**：
- `pytest.skip("scip_pb2.py not generated yet")` → 见 L1
- chunk/relation 字段不符合预期 → 多半是 `_process_document` 或 `_build_global_definition_index` 的逻辑改动；对照 `tests/test_scip_clang_backend.py::_build_minimal_index` 看入参输出对应关系

---

## L4 — config / CLI / 调用链 plumbing 单测

确认 backend 选择能正确穿透 `IndexingConfig` → CLI `--backend` → `build_kernel_index` → 调用链查询层。

```bash
python3 -m pytest tests/test_config_indexing_backend.py tests/test_call_chain_call_site.py -v 2>&1 | tail -30
```

**期望**：`15 passed`。覆盖：

- `normalize_raw_config` 默认 `backend=clangd`，scip-clang 子块默认值正确
- 显式 `backend: scip-clang` 与 `scip_clang.binary/timeout_sec/extra_args/keep_scip_file` 被吃进 `AppConfig.indexing`
- `retrieve_call_chain` 在 scip-clang 行上返回 `call_site_path / call_site_line / call_site_col`
- 在 clangd-origin 行上不返回 call-site 字段
- markdown / 文本输出格式化器对 call-site 后缀的处理一致

**失败模式**：
- `pydantic.ValidationError` → `IndexingConfig` 或 `ScipClangConfig` 的字段类型与测试期望不符
- call-chain 测试挂 → `llamaindex_pipeline.retrieve_call_chain` 改动后没同步 schema

> **离线层到此打底完成。45 个测试全过则进入在线层。**

---

## L5 — 玩具 C 项目端到端（首次用真二进制）

最小可信端到端：用 ~3 个 C 源文件 + 1 个 `compile_commands.json` 让 scip-clang 真的跑一次，并把输出推到内存里的 `CodeIndex`。

### 5.1 准备 fixture

```bash
mkdir -p /tmp/scip-smoke/src && cd /tmp/scip-smoke
cat > src/helper.h <<'EOF'
int helper(int x);
EOF
cat > src/helper.c <<'EOF'
#include "helper.h"
int helper(int x) { return x + 1; }
EOF
cat > src/main.c <<'EOF'
#include "helper.h"
int main(void) {
    return helper(41);
}
EOF
cat > compile_commands.json <<EOF
[
  {"directory":"$(pwd)","file":"src/helper.c","command":"clang -c -I src src/helper.c -o /tmp/helper.o"},
  {"directory":"$(pwd)","file":"src/main.c","command":"clang -c -I src src/main.c -o /tmp/main.o"}
]
EOF
```

### 5.2 直接调 scip-clang 验证二进制可用

```bash
scip-clang --compdb-path=compile_commands.json -o /tmp/smoke.scip
ls -lh /tmp/smoke.scip          # 期望几 KB 量级，非 0
```

**期望**：进程退出码 0；`/tmp/smoke.scip` 非空。`stderr` 可能含 `failed` 字样（少数 TU 失败）— 这是正常的，scip-clang 的设计是 skip 失败 TU 继续。

### 5.3 用 backend 直接解析这个真 `.scip`

```bash
cd /home/user/hm-kernel-llm-opt    # 切回项目根
python3 -c "
import sys
sys.path.insert(0, 'src')
# 绕开 hmopt.indexing 顶层 import 链（拉 llama_index/neo4j）
import importlib, importlib.util
from pathlib import Path

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

_load('m_pb2', 'src/hmopt/indexing/_generated/scip_pb2.py')
sys.modules.setdefault('hmopt.indexing._generated', type(sys)('p'))
sys.modules['hmopt.indexing._generated'].scip_pb2 = sys.modules['m_pb2']
# 上面是把 scip_pb2 挂到 backend 期望的 import 路径上的最小 hack
# 真实代码里 ``pip install -e .`` 后这一段都不需要
import hmopt.indexing.backends.scip_clang as sc
from hmopt.indexing.backends.scip_clang import ScipClangBackend, ScipClangConfig

idx = ScipClangBackend(ScipClangConfig(keep_scip_file=False))._parse_scip(
    Path('/tmp/smoke.scip'), Path('/tmp/scip-smoke'), {}
)
print('chunks=', len(idx.chunks), 'relations=', len(idx.relations))
for c in idx.chunks:
    print(' chunk:', c.symbol_name, c.kind, c.path, c.start_line, '-', c.end_line)
for r in idx.relations:
    print(' rel:', r.kind, r.src_name, '->', r.dst_name,
          'cs=', r.call_site_path, r.call_site_line, r.call_site_col)
"
```

> 上面的 sys.modules hack 是为了在不 pip install 的环境下能直接跑；正常开发环境里 `pip install -e .` 之后只需 `from hmopt.indexing.backends.scip_clang import ScipClangBackend`。

**期望**：
- `chunks` 至少 2（`helper`、`main`）
- `relations` 至少 1（`main` → `helper`，`kind="calls"`）
- 这条 relation 上 `call_site_path` 形如 `src/main.c`，`call_site_line=3`，`call_site_col` 约 12（指向 `helper(` 的 `h`）

**失败模式**：
- `ScipClangError: scip-clang binary not on PATH` → 安装：`curl -sL https://github.com/sourcegraph/scip-clang/releases/latest/download/scip-clang-x86_64-linux -o /usr/local/bin/scip-clang && chmod +x /usr/local/bin/scip-clang`
- `scip-clang produced no output (exit=...)` → 看 `stderr tail`；常见原因是 `compile_commands.json` 里 `directory` 不是绝对路径，或 `-c` 缺失
- chunks 数为 0 → scip-clang 跑了但没解析出任何定义，多半是头文件路径错；用 `scip-clang --compdb-path=... -o /tmp/x.scip --` 加 `-v` 查 driver 实际命令

### 5.4 通过 CLI 入口闭环（pip install 之后）

```bash
pip install -e .
HMOPT_REPO_PATH=/tmp/scip-smoke \
  hmopt index-kernel \
    --repo-path /tmp/scip-smoke \
    --compile-commands-dir /tmp/scip-smoke \
    --backend scip-clang
```

**期望**：终端打印 `Kernel code index built (backend=scip-clang)`；如果 `configs/app.yaml` 里启用了 Neo4j/向量库，写入应当成功。

---

## L6 — 真实内核子树端到端

挑一个范围适中的子树（建议从 `kernel/sched/` 或 `kernel/workqueue.c` 这类单文件开始），有现成 `compile_commands.json`。

### 6.1 预飞行

```bash
KSRC=/path/to/hm-verif-kernel
CCJ=/path/to/compile_commands_dir   # 通常是 kernel build 输出目录

ls "$CCJ/compile_commands.json"     # 必须存在
jq 'length' "$CCJ/compile_commands.json"   # 期望 ≥ 1，看条目数
```

### 6.2 单跑 scip-clang（不入库）观察规模

```bash
time scip-clang --compdb-path="$CCJ/compile_commands.json" -o /tmp/kernel.scip
ls -lh /tmp/kernel.scip
```

**经验值**：
- 小子树（< 100 TUs）：< 1 min，`.scip` 几 MB
- 中等子树（hyperhold / reclaim 完整子系统）：5–30 min，`.scip` 几十 MB
- 全内核：1–2 h，~375 MB（与 plan §"Risks & mitigations" 一致）

**失败模式**：
- 大量 `failed` 行 + `.scip` 比预期小很多 → 检查 toolchain：`scip-clang` 与 kernel 实际使用的 clang 主版本是否一致；可以通过 `--cdb-flags='-resource-dir=...'` 强制
- 进程被 OOM 杀 → 内存峰值大约是 `.scip` 体积的 3–4 倍；中等子树需要 ≥ 16 GB

### 6.3 走 hmopt 全流程

```bash
hmopt index-kernel \
  --repo-path "$KSRC" \
  --compile-commands-dir "$CCJ" \
  --backend scip-clang 2>&1 | tee /tmp/scip-index.log
```

**关键日志行**（可在 log 里 grep 出来）：

```
ScipClangBackend.build complete: chunks=N relations=M duration_s=T failed_tus=K
```

并核对 backend 自动写入的 diagnostics：

```python
python3 -c "
# 这里假设有一个落盘 diagnostics 的位置（看 llamaindex_pipeline.build_kernel_index 决定）
# 否则就在调用处临时 print(index.diagnostics)
"
```

### 6.4 入库后查图（如果配了 Neo4j）

```cypher
// 边数总览
MATCH ()-[r]->() RETURN r.backend_origin, count(r);

// scip-clang 独有字段覆盖率
MATCH ()-[r {backend_origin: "scip-clang"}]->()
RETURN
  count(r) AS total,
  sum(CASE WHEN r.call_site_line IS NOT NULL THEN 1 ELSE 0 END) AS with_call_site,
  sum(CASE WHEN r.syntax_kind IS NOT NULL THEN 1 ELSE 0 END) AS with_syntax_kind;
```

**期望**：
- `with_call_site / total ≥ 0.95`（plan §Phase 3 exit criteria）
- `with_syntax_kind / total ≥ 0.95`

---

## L7 — clangd ↔ scip-clang A/B 等价校验

只有这层通过，才能宣告 scip-clang 可作为默认 backend。

### 7.1 同一目标跑两次

```bash
# 把 Neo4j 数据库切到 graph_clangd
hmopt index-kernel --repo-path "$KSRC" --compile-commands-dir "$CCJ" \
                   --backend clangd
# 导出
neo4j-admin database dump --to-path=/tmp/dumps graph_clangd

# 切到 graph_scip
hmopt index-kernel --repo-path "$KSRC" --compile-commands-dir "$CCJ" \
                   --backend scip-clang
neo4j-admin database dump --to-path=/tmp/dumps graph_scip
```

### 7.2 边集合重合度

```cypher
// 在同一个 Neo4j 实例两个 db 间用 apoc 或者用脚本：
// 拉两边的 (src_id, dst_id, kind) 三元组成 set，算交集 / 较小者
```

**期望**（plan §Phase 3 exit criteria）：

| 指标 | 阈值 |
|---|---|
| 三元组重合度 `|A ∩ B| / min(|A|, |B|)` | ≥ 80% |
| 仅 clangd 独有 | 详查；通常是 LSP `references` 覆盖到的 scip-clang 不视作 call 的句法构造 |
| 仅 scip-clang 独有 | 详查；通常是 macro 展开后才能看见的 call、function-pointer take |
| call-site 字段非空率（scip-clang 侧） | ≥ 95% |

### 7.3 产物：A/B 报告

把上面的对比写到 `docs/scip_clang_eval.md`，包括：

- 目标 target + commit hash
- 两次跑批的 `duration_s`、`chunks`、`relations`、`failed_tus`
- 三元组重合度具体数字
- 差集中前 20 条样本（人工审）
- 结论：是否切换默认 backend；若切换，何时切

> A/B 报告就是 plan §Phase 5 的"如果 favorable: 切换默认 backend" 的判断依据，必须落档。

---

## 失败排查 Cookbook

| 症状 | 第一时间检查 | 命令 |
|---|---|---|
| `scip-clang: command not found` | 二进制装没装、PATH 对不对 | `which scip-clang; echo $PATH` |
| `scip-clang produced no output` | compile_commands.json 路径与 `directory` 字段 | `jq '.[0]' compile_commands.json` |
| `failed_tu_count` 很大 | scip-clang 与系统 clang 主版本错配 | `scip-clang --version; clang --version` |
| `Couldn't build proto file ... DescriptorPool` | protobuf 运行时与生成代码不匹配 | `pip show protobuf; head -5 src/hmopt/indexing/_generated/scip_pb2.py` |
| chunks 数为 0 但 `.scip` 不空 | 全是 reference、没有 Definition；查头文件搜索路径 | scip-clang 加 `-- -v` 看 driver 命令 |
| Neo4j 写入慢 / OOM | `.scip` 体量大 + 一次性 load | `keep_scip_file: true` 落盘后分段 load |
| call_site_line 都是 None | 走的还是 clangd backend（`backend_origin` 没变） | 看 `cli` / `app.yaml` 的 `backend` 字段是否被覆盖到 |
| A/B 重合度 < 80% | 多半是头文件 vs cpp 实现的归位策略不同 | 对差集按 `kind` 分桶，看是哪类 relation 失配 |

---

## 退出标准（per-level Exit Criteria 汇总）

- ✅ **L0 通过**：工具链就位
- ✅ **L1 通过**：`scip_pb2.Index` 可往返
- ✅ **L2 通过**：`tests/test_scip_translation.py` 27 个测试全过
- ✅ **L3 通过**：`tests/test_scip_clang_backend.py` 3 个测试全过
- ✅ **L4 通过**：`tests/test_config_indexing_backend.py` + `tests/test_call_chain_call_site.py` 15 个测试全过
- ⏳ **L5 通过**：玩具 C 项目跑出真 `.scip`，backend 解析出 `main → helper` 调用边带 call-site
- ⏳ **L6 通过**：真实内核子树 `hmopt index-kernel --backend scip-clang` 完成，Neo4j 里 scip-only 字段覆盖率 ≥ 95%
- ⏳ **L7 通过**：A/B 重合度 ≥ 80%，落档 `docs/scip_clang_eval.md`

> 当前仓库 commit 状态：**L0–L4 已通过**（45 个离线测试全过）。L5–L7 需要在装有 scip-clang 二进制的环境里推进。

---

## 一键脚本骨架（可选下一步）

可以把 L1–L4 串成一个本地 smoke 脚本（不在本计划交付范围内，仅给出思路）：

```bash
# scripts/verify_scip_offline.sh —— 建议下一步落地
set -euo pipefail
echo "[L1] protobuf import & round-trip"
python3 -c "..."   # L1 命令
echo "[L2-L4] pytest offline layers"
python3 -m pytest tests/test_scip_translation.py \
                  tests/test_scip_clang_backend.py \
                  tests/test_config_indexing_backend.py \
                  tests/test_call_chain_call_site.py -q
echo "✓ scip-clang offline layers all green"
```

---

*本文档为测试验证步骤，不修改任何产品代码。*
