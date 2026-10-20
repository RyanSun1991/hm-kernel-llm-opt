"""Read-only research templates adapted from the reviewed planning appendix.

These are questions for investigation, not executable matchers, discovered
defects or activated Patterns. No seed supplies invented historical evidence.
"""

from __future__ import annotations

from copy import deepcopy

_SEEDS = (
    {
        "id": "P001",
        "title": "循环不变量复用研究",
        "kind": "optimization",
        "mechanism": "hoist-invariant",
        "anchor_hint": "循环中重复访问同一字段或表达式，仅作为待核实线索。",
        "required_proofs": [
            "证明循环内及被调用函数没有通过别名改变该值。",
            "证明并发写入、volatile 访问和生命周期不要求每次重新观察。",
            "比较实际构建的机器码，确认编译器未完成相同变换。",
        ],
        "negative_examples": ["状态字段由另一执行者更新。", "循环调用经别名修改字段。"],
        "acceptance_kind": "performance",
        "metric_guidance": "根据瓶颈冻结 IC、时延或访问成本；功能与并发正确性作为必要约束。",
        "risk": "unresolved",
        "default_lane": "workbench",
    },
    {
        "id": "P002",
        "title": "分配、用户复制与 MMIO 的分别批处理研究",
        "kind": "optimization",
        "mechanism": "batch-coalesce",
        "anchor_hint": "循环中的 alloc、copy_to_user 或 MMIO 调用；三种机制分别立案。",
        "required_proofs": [
            "分配变体：证明对象大小、所有权、释放时机、失败路径和内存压力下语义保持。",
            "用户复制变体：证明访问边界、缺页、部分复制、错误返回和用户内存并发变化的语义保持。",
            "MMIO 变体：核对设备与架构契约，证明访问宽度、顺序、屏障及每次访问可见副作用保持。",
            "不得把某一变体的成功结论迁移为另外两种机制的可批量化证明。",
        ],
        "negative_examples": [
            "寄存器读操作本身会清除设备状态。",
            "合并复制改变部分成功及错误返回语义。",
            "扩大分配导致高阶分配失败或释放时机变化。",
        ],
        "acceptance_kind": "performance",
        "metric_guidance": "分别绑定分配成本、复制路径成本或设备访问时延；测量吞吐、尾延迟及内存护栏。",
        "risk": "high",
        "default_lane": "workbench",
    },
    {
        "id": "P003",
        "title": "锁保护范围与同步成本研究",
        "kind": "optimization",
        "mechanism": "lock-scope-analysis",
        "anchor_hint": "循环中加锁提示需要分析临界区；不能证明锁冗余。",
        "required_proofs": [
            "列明锁保护的不变量、读写参与者、锁顺序及中断/抢占上下文。",
            "任何同步变更必须给出替代的互斥、可见性及生命周期证明。",
            "通过独立并发评审与有针对性的竞争、死锁和压力验证后再评估性能。",
            "禁止由种子、词法次数或热点结果自动删除锁。",
        ],
        "negative_examples": [
            "持锁时间短，但仍保护共享对象生命周期。",
            "测试期间未出现竞争，生产上下文仍存在并发访问。",
        ],
        "acceptance_kind": "performance",
        "metric_guidance": "以已证实的锁等待或尾延迟为指标，同时保留同步正确性验证门。",
        "risk": "high",
        "default_lane": "workbench",
    },
    {
        "id": "P004",
        "title": "重复原子观察的语义研究",
        "kind": "optimization",
        "mechanism": "atomic-observation-analysis",
        "anchor_hint": "同一对象的多次原子读取可能用于观察变化，不按读取次数判定冗余。",
        "required_proofs": [
            "证明算法允许复用同一次观察值，并检查其他执行者的更新协议。",
            "核对具体 API、架构、编译配置的原子性与内存顺序契约。",
            "检查轮询、退出、重试和进度保证；禁止自动删除或普通化原子访问。",
        ],
        "negative_examples": [
            "第二次读取用于识别生产者刚发布的数据。",
            "重试循环必须持续重新观察共享状态。",
        ],
        "acceptance_kind": "performance",
        "metric_guidance": "在语义证明后测量实际机器码与原子访问成本；收益不能替代进度和并发验证。",
        "risk": "high",
        "default_lane": "workbench",
    },
    {
        "id": "P005",
        "title": "循环分支不变性研究",
        "kind": "optimization",
        "mechanism": "branch-invariance-analysis",
        "anchor_hint": "循环中的分支谓词疑似不变；结构分析无法证明时必须保留 unknown。",
        "required_proofs": [
            "证明谓词及其依赖在整个作用域内不变，包括别名与调用副作用。",
            "证明提前求值不会改变异常、错误处理或副作用顺序。",
            "检查编译器已有处理、分支预测与代码尺寸变化。",
        ],
        "negative_examples": [
            "谓词依赖循环内递增计数或被修改的配置。",
            "提前执行条件表达式触发原本不会发生的访问。",
        ],
        "acceptance_kind": "performance",
        "metric_guidance": "根据实际瓶颈选择分支失误、IC 或时延，同时检查代码尺寸和冷热路径护栏。",
        "risk": "unresolved",
        "default_lane": "workbench",
    },
    {
        "id": "P006",
        "title": "间接调用目标稳定性研究",
        "kind": "optimization",
        "mechanism": "indirection-target-analysis",
        "anchor_hint": "循环中的函数指针调用，需要核实目标稳定性与替换机制。",
        "required_proofs": [
            "覆盖目标注册、注销、动态替换、配置选择和模块生命周期。",
            "证明目标在优化作用域中保持稳定，保留接口允许的多态语义。",
            "检查实际机器码、编译器去虚拟化及目标平台调用成本。",
        ],
        "negative_examples": ["运行时可替换操作表。", "不同配置或设备使用不同函数实现。"],
        "acceptance_kind": "performance",
        "metric_guidance": "测量调用路径 IC 或时延，并覆盖不同目标与配置的功能行为。",
        "risk": "unresolved",
        "default_lane": "workbench",
    },
    {
        "id": "P007",
        "title": "热点短函数的实际内联行为研究",
        "kind": "optimization",
        "mechanism": "inline-callee-investigation",
        "anchor_hint": "短函数在热点中被调用；源码未标 inline 不证明机器码未内联。",
        "required_proofs": [
            "检查目标构建配置、优化级别、LTO、汇编或编译器优化报告。",
            "确认真实调用开销尚存在且对主瓶颈有贡献。",
            "比较代码尺寸、指令缓存及其他调用点，不能以源码行数作为成本模型。",
        ],
        "negative_examples": [
            "未标 inline 的函数已被编译器自动内联。",
            "强制内联扩大热代码并恶化指令缓存。",
        ],
        "acceptance_kind": "performance",
        "metric_guidance": "在机器码证据基础上度量热点时延或 IC，代码尺寸及指令缓存作为护栏。",
        "risk": "unresolved",
        "default_lane": "workbench",
    },
    {
        "id": "P008",
        "title": "有证据的访存预取研究",
        "kind": "optimization",
        "mechanism": "prefetch-investigation",
        "anchor_hint": "热点中的指针链或顺序访问；热点比例本身不能证明访存延迟瓶颈。",
        "required_proofs": [
            "提供缓存未命中、访存等待或等价的瓶颈证据。",
            "核实目标地址、对象生命周期、架构指令语义与访问安全。",
            "评估有效预取距离、缓存污染、带宽和不同工作负载下行为。",
        ],
        "negative_examples": [
            "路径由锁等待主导而非访存等待。",
            "预取占用带宽并逐出更有价值的数据。",
        ],
        "acceptance_kind": "performance",
        "metric_guidance": "以访存相关时延或吞吐为主指标，带宽、缓存污染与整机负载作为护栏。",
        "risk": "high",
        "default_lane": "workbench",
    },
    {
        "id": "P009",
        "title": "重算与缓存的成本和失效研究",
        "kind": "optimization",
        "mechanism": "recompute-vs-cache",
        "anchor_hint": "重复读取或调用提示成本分析；计数不能决定缓存或重算方向。",
        "required_proofs": [
            "明确状态可变性、失效时机、缓存键、所有权及并发同步。",
            "分别测量重算、存取、失效与维护成本。",
            "评估空间占用、缓存局部性、错误路径和最坏情况下资源压力。",
        ],
        "negative_examples": ["缺少失效协议导致使用过期结果。", "缓存维护和访问比廉价重算更昂贵。"],
        "acceptance_kind": "performance",
        "metric_guidance": "以已分类瓶颈的时延、IC 或内存成本为主指标，另一类资源成本作为护栏。",
        "risk": "unresolved",
        "default_lane": "workbench",
    },
    {
        "id": "D001",
        "title": "持自旋类锁路径的可睡眠调用诊断",
        "kind": "diagnostic",
        "mechanism": "sleep-under-spinlock",
        "anchor_hint": "持锁路径调用疑似可睡眠操作；必须核对具体锁类型、配置与调用路径。",
        "required_proofs": [
            "解析实际锁 API、构建配置及执行上下文，确认禁止睡眠的契约。",
            "提供可达调用路径及具体可睡眠行为的证据，而非函数名猜测。",
            "建立基线复现及修复后正确性检查，保留其他锁与资源约束。",
        ],
        "negative_examples": [
            "实际锁类型或配置允许相应阻塞行为。",
            "疑似调用路径在持锁条件下不可达。",
        ],
        "acceptance_kind": "correctness",
        "metric_guidance": "主验收为复现、锁约束及针对性正确性检查；lmbench 或时延结果只能作为性能护栏。",
        "risk": "high",
        "default_lane": "workbench",
    },
    {
        "id": "D002",
        "title": "循环屏障的顺序契约诊断",
        "kind": "diagnostic",
        "mechanism": "barrier-in-loop",
        "anchor_hint": "循环中出现内存屏障仅提示核对顺序契约，不能据位置或耗时判定冗余。",
        "required_proofs": [
            "列明屏障两侧的读写、通信参与者及必须建立的可见性和顺序关系。",
            "核对目标 API、编译器及架构内存模型，必要时建立 litmus 或等价顺序测试。",
            "禁止自动删除、合并或弱化屏障；任何修改须经过独立并发审阅。",
        ],
        "negative_examples": [
            "每轮发布都依赖该屏障建立顺序。",
            "单一架构上未复现不能证明弱顺序架构上正确。",
        ],
        "acceptance_kind": "correctness",
        "metric_guidance": "顺序关系和并发正确性是主验收，屏障成本仅是次级性能观察。",
        "risk": "high",
        "default_lane": "workbench",
    },
    {
        "id": "D003",
        "title": "复杂锁路径检查",
        "kind": "diagnostic",
        "mechanism": "complex-lock-path-inspection",
        "anchor_hint": "存在锁调用且分支较多，提示路径检查；不能证明缺少解锁。",
        "required_proofs": [
            "建立控制流、锁状态及资源所有权分析，覆盖返回、错误、跳转和跨函数释放。",
            "区分锁转移、条件持锁、辅助函数解锁及有意保持锁的 API 契约。",
            "只有发现具体可达且违反契约的路径后，才生成缺锁或漏解锁诊断候选。",
        ],
        "negative_examples": [
            "分支很多但所有路径均正确释放锁。",
            "解锁由被调用函数或明确的调用方契约负责。",
        ],
        "acceptance_kind": "correctness",
        "metric_guidance": "主验收为路径证据、基线复现与修复后锁状态检查；性能只作为护栏。",
        "risk": "high",
        "default_lane": "workbench",
    },
)


def seed_catalog() -> dict:
    """Return a fresh research-only catalog; reading it never imports a Pattern."""
    return {
        "schema_version": 1,
        "status": "draft_research_templates",
        "provenance": "Design appendix reviewed against implementation constraints; no observed defects or source evidence are asserted.",
        "activation_allowed": False,
        "automatic_execution": False,
        "conversion_requirements": [
            "Collect real persisted source evidence and inspect positive and negative examples.",
            "Establish applicability proofs and select a frozen correctness or performance acceptance policy.",
            "Create a separate draft Pattern with real source IDs; independent curation is required before activation.",
        ],
        "templates": deepcopy(list(_SEEDS)),
    }
