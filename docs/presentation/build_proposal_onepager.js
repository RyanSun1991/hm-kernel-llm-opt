/* Team Skill Hub — one-page proposal slide (CN primary + EN variant). */
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const {
  FaTriangleExclamation, FaArrowsRotate, FaLightbulb, FaChartLine,
} = require("react-icons/fa6");

// ---- palette ----
const INK = "1A202C", NAVY = "2B368C", TEAL = "0E9488", TEALD = "0B7A70";
const AMBER = "C77E14", RED = "C03A3A", MUTED = "5B657A";
const PANEL = "F1F4FA", NAVYBG = "E9EDF9", TEALBG = "E1F2F0", PURBG = "EAE8F8";
const PUR = "4A44A8", REDBG = "FBEDED", LINE = "C9D0DE", WHITE = "FFFFFF";

async function iconPng(Icon) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(Icon, { color: "#FFFFFF", size: 256 }));
  const buf = await sharp(Buffer.from(svg))
    .resize(256, 256, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ---------------------------------------------------------------- content --
const CN = {
  font: "Microsoft YaHei",
  kicker: "鸿蒙内核智能优化 · AGENT 长期记忆与质量治理基础设施",
  titleA: "面向鸿蒙内核智能优化的团队级自进化经验中枢  ",
  titleB: "Team Skill Hub",
  titleSizeA: 21, titleSizeB: 15,
  chip1: "从「一次性能力」到「可复利资产」",
  chip1Size: 10.5,
  chip2: "One-shot Capability → Compounding Assets",
  subtitle: "双资产双引擎治理 × 全自动闭环 —— 让团队优化经验像「私有 npm 包」一样可积累、可验证、可审计、可复利",
  painH: "场景问题",
  painT: "规模化的真瓶颈：不是单次优化，而是经验复利",
  pains: [
    { t: "经验孤岛", d: "哪个招式有效、哪里有坑、热函数怎么改、哪种验证不可信 —— 只存在工程师个人本地，无法跨人、跨项目、跨时间复用", red: false },
    { t: "重复踩坑 · 无法传承", d: "团队反复探索同一方向、重蹈同一坑；专家经验带不走，新人与新 Agent 每次从零开始，优化边际效率不升反降", red: false },
    { t: "「简单共享」= 两大致命风险", d: "① 知识漂移：经验自相矛盾、随时间失效；② 反馈自增强：均值变好、个别场景被悄悄做坏 —— 业界 mem0 / Zep 均缺团队级治理，规模越大风险越大", red: true },
  ],
  painC: "需要的不是又一个「记忆库」，而是带质量治理、可自进化的团队级经验基础设施",
  solH: "方案",
  solT: "版本化经验中枢 × 双资产双引擎 × 「消费 → 蒸馏 → 门控 → 发布 → 再消费」全自动闭环",
  hubTitle: "hm-skill-hub · 团队中央经验仓 —— semver 发布 + lockfile 钉版 · 像「私有 npm」一样消费 · 可回滚 · 全程可审计",
  hubTitleSize: 11,
  engineBodySize: 8.5,
  engA: { tag: "Knowledge 知识资产（事实 · 教训 · 坏招）", sub: "引擎 A · 治理型合并",
    body: "只追加 + 七路关系分类（重复 / 矛盾 / 过时 / 条件 / 泛化 / 漂移 / 口径）· 双时态墓碑，永不物删" },
  engB: { tag: "Skills 技能资产（做法 · 流程）", sub: "引擎 B · 竞争式进化",
    body: "SkillOpt 有界编辑：留出 eval「严格变好且零回归」才接受 + Pareto 前沿防多人编辑塌缩" },
  loops: [
    { n: "1", t: "消费 · 自动挂载", d: "流水线 research→…→test 按阶段注入 Hub 上下文；MCP 接入 · 6 个接入点 · 断连静默降级不阻塞" },
    { n: "2", t: "收口 · 自动蒸馏", d: "run 收口双段抽取（确定性规则 + LLM 显著性）→ schema 合规 Tier-1 候选，自动附出处与证据" },
    { n: "3", t: "门控 · 策展入库", d: "CI 五道门（lint · 脱敏 · 去重 · eval-gate）+ 双人评审；≥2 独立实例自动检测毕业为 technique 技能" },
  ],
  relN: "4",
  relLabel: "发布回灌：nightly 七步作业 → semver + scorecard → broadcast 自动钉版",
  promoteLabel: "过三道质量门 → 入库",
  innoH: "核心创新与差异化",
  innos: [
    { t: "双资产 · 双引擎", d: "知识与技能严格分治、各配治理引擎，根除「漂移」与「坏经验覆盖」—— mem0 / Zep 缺失的团队治理层" },
    { t: "七路分类 × 双时态", d: "矛盾消解永不物删：superseded 墓碑 + 有效期，全程可审计；分类基准 48 例准确率 100%、误删 0" },
    { t: "eval-gate 反喂安全", d: "技能 = 可训练外部参数：改动须「严格变好且零回归」才入库；Pareto 留互补候选，根治自增强跑偏" },
    { t: "全自动闭环工程化", d: "收口蒸馏 → CI 门 → nightly 七步 → semver 发布 → 钉版回灌全自动；人只握晋升审批权" },
  ],
  fxH: "预期效果",
  fxT: "经验从「个人消耗品」变成「团队复利资产」",
  fx: [
    { t: "开箱即会", d: "新人 / 新 Agent 首次部署即携带全队经验，重复探索与重复踩坑显著减少" },
    { t: "专家经验资产化", d: "高手经验沉淀为质量门保护的团队资产，不再随人员流动而流失" },
    { t: "复利式提升", d: "经验跨成员、跨迭代持续累积，优化吞吐与命中率随使用不断上升" },
    { t: "通用自进化底座", d: "内存底噪 / 指令数 / 功耗 / 编译器后端等任意 Agent 优化场景即插即用" },
  ],
  factsLead: "落地验证",
  facts: "全链路已实现、离线可复现：154 项测试 · 7 份 Schema · 5 道 CI 门 · 3 个 MCP 工具 · 6 个接入点 · 16 项机制词表 · 检索 recall@5=1.0（26 查询）· 七路分类 48/48 · 优化门 0.67→1.00",
  factsSize: 8,
};

const EN = {
  font: "Arial",
  kicker: "HARMONY KERNEL INTELLIGENT OPTIMIZATION · LONG-TERM MEMORY & QUALITY GOVERNANCE FOR AGENTS",
  titleA: "Team-Level Self-Evolving Experience Hub  ",
  titleB: "Team Skill Hub",
  titleSizeA: 20, titleSizeB: 14,
  chip1: "One-Shot Capability → Compounding Assets",
  chip1Size: 9.5,
  chip2: "Long-term memory infra for optimization agents",
  subtitle: "Dual engines × fully automated loop — team experience as a private npm package: accumulable · verifiable · auditable · compounding",
  painH: "Scenario & Pain",
  painT: "At scale the bottleneck is compounding experience, not one-shot wins",
  pains: [
    { t: "Experience silos", d: "Effective moves, known pits, hot-function fixes, unreliable validations — all trapped in one engineer's workspace; never reused across people, projects, or time", red: false },
    { t: "Repeated pitfalls, no inheritance", d: "Teams re-explore the same directions and re-step into the same traps; every new member or agent restarts from zero — marginal efficiency declines", red: false },
    { t: "Naive sharing = two fatal risks", d: "① Knowledge drift — entries contradict and decay; ② self-reinforcing feedback — averages improve, edge cases silently regress; mem0 / Zep lack team-level governance", red: true },
  ],
  painC: "Needed: not another memory store — a governed, self-evolving team experience infrastructure",
  solH: "Solution",
  solT: "Versioned hub × dual engines × fully automated “consume → distill → gate → release → re-consume” loop",
  hubTitle: "hm-skill-hub · central team hub — semver releases + lockfile pinning · a “private npm” for experience · rollbackable · auditable",
  hubTitleSize: 10.5,
  engineBodySize: 8,
  engA: { tag: "Knowledge (facts · lessons · bad plans)", sub: "Engine A · governed merge",
    body: "Append-only + 7-way relation classes (dup / contradiction / temporal / conditional / subsumption / drift / basis) · bitemporal tombstones, never delete" },
  engB: { tag: "Skills (procedures · playbooks)", sub: "Engine B · competitive evolution",
    body: "SkillOpt bounded edits — accepted only if strictly better with zero regression on held-out evals; Pareto frontier keeps complementary candidates" },
  loops: [
    { n: "1", t: "Consume · auto-mount", d: "Staged pipeline (research→…→test) gets per-stage “Hub context” injection; MCP-integrated · 6 touchpoints · fail-silent degradation" },
    { n: "2", t: "Distill at close-out", d: "Two-stage extraction (deterministic rules + LLM salience) → schema-valid Tier-1 candidates with provenance & evidence" },
    { n: "3", t: "Gate · curate", d: "5 CI gates (lint · redact · dedup · eval-gate) + dual review; ≥2 independent instances auto-graduate into a technique skill" },
  ],
  relN: "4",
  relLabel: "Release & feed back: nightly 7-step job → semver + scorecard → broadcast re-pins — next run carries the whole team's experience",
  promoteLabel: "3 quality gates → land",
  innoH: "Core Innovations & Differentiation",
  innos: [
    { t: "Dual asset · dual engine", d: "Knowledge and skills strictly separated, each with its own governance engine — the team-governance layer mem0 / Zep lack" },
    { t: "7-way classes × bitemporal", d: "Conflict resolution never deletes history: superseded tombstones + validity windows; 48-case benchmark: 100% accuracy, 0 false-deletes" },
    { t: "Eval-gated feedback safety", d: "Skill text = trainable external parameters: only strictly-better, zero-regression edits land; Pareto kills self-reinforcement" },
    { t: "Fully automated loop", d: "Close-out distill → CI gates → nightly 7 steps → semver release → re-pin, all automated; humans keep promotion approval" },
  ],
  fxH: "Expected Impact",
  fxT: "Experience: from personal consumable to compounding team asset",
  fx: [
    { t: "Day-one mastery", d: "New members & agents deploy with the whole team's experience — far less re-exploration" },
    { t: "Experts → assets", d: "Expert know-how becomes gate-protected team assets that survive turnover" },
    { t: "Compounding gains", d: "Throughput and hit-rate keep rising as experience accrues across members & iterations" },
    { t: "Universal memory base", d: "Plug-in for any agent-driven optimization: memory noise floor / instructions / power / compiler backends" },
  ],
  factsLead: "Implemented",
  facts: "End-to-end & reproducible offline: 154 tests · 7 schemas · 5 CI gates · 3 MCP tools · 6 touchpoints · 16-mechanism vocab · retrieval recall@5 = 1.0 (26 queries) · classifier 48/48 · skill-opt 0.67 → 1.00",
  factsSize: 8,
};

// ---------------------------------------------------------------- layout ---
async function main() {
  const icons = {
    warn: await iconPng(FaTriangleExclamation),
    loop: await iconPng(FaArrowsRotate),
    bulb: await iconPng(FaLightbulb),
    chart: await iconPng(FaChartLine),
  };

  const pres = new pptxgen();
  pres.defineLayout({ name: "W", width: 13.333, height: 7.5 });
  pres.layout = "W";

  const M = 0.32, W = 13.333;
  const RX = 3.58, RW = W - M - RX; // right zone

  function render(S) {
    const F = S.font;
    const s = pres.addSlide();
    s.background = { color: WHITE };

    // ---- header ----
    s.addText(S.kicker, { x: M, y: 0.18, w: 9.6, h: 0.22, fontFace: F, fontSize: 9.5,
      bold: true, color: TEAL, charSpacing: 1.5, margin: 0, valign: "middle" });
    s.addText([
      { text: S.titleA, options: { color: NAVY, fontSize: S.titleSizeA } },
      { text: S.titleB, options: { color: TEAL, fontSize: S.titleSizeB } },
    ], { x: M, y: 0.42, w: 9.65, h: 0.46, fontFace: F, bold: true,
      margin: 0, valign: "middle" });
    s.addText(S.subtitle, { x: M, y: 0.94, w: 9.65, h: 0.24, fontFace: F,
      fontSize: 10.5, color: MUTED, margin: 0, valign: "middle" });
    // top-right positioning chip
    s.addShape(pres.ShapeType.roundRect, { x: 10.05, y: 0.26, w: 2.95, h: 0.78,
      fill: { color: NAVY }, line: { type: "none" }, rectRadius: 0.08 });
    s.addText([
      { text: S.chip1, options: { fontSize: S.chip1Size, bold: true, color: WHITE, breakLine: true } },
      { text: S.chip2, options: { fontSize: 7.5, color: "C7CFF2" } },
    ], { x: 10.11, y: 0.26, w: 2.83, h: 0.78, fontFace: F, align: "center",
      valign: "middle", margin: 0 });

    // ---- left column: pains ----
    s.addShape(pres.ShapeType.ellipse, { x: M, y: 1.19, w: 0.30, h: 0.30,
      fill: { color: RED }, line: { type: "none" } });
    s.addImage({ data: icons.warn, x: M + 0.07, y: 1.26, w: 0.16, h: 0.16 });
    s.addText([
      { text: S.painH, options: { fontSize: 14, bold: true, color: NAVY, breakLine: true } },
      { text: S.painT, options: { fontSize: 8.5, color: MUTED } },
    ], { x: M + 0.40, y: 1.13, w: 2.75, h: 0.62, fontFace: F, margin: 0, valign: "top",
      lineSpacingMultiple: 1.05 });

    const painYs = [1.80, 2.92, 4.04];
    S.pains.forEach((p, i) => {
      const y = painYs[i], h = 1.02;
      s.addShape(pres.ShapeType.roundRect, { x: M, y, w: 3.10, h,
        fill: { color: p.red ? REDBG : PANEL },
        line: { color: p.red ? "E3B8B8" : LINE, width: 0.75 }, rectRadius: 0.05 });
      s.addText([
        { text: p.t, options: { fontSize: 10, bold: true, color: p.red ? RED : NAVY, breakLine: true, paraSpaceAfter: 3 } },
        { text: p.d, options: { fontSize: 8.5, color: INK } },
      ], { x: M + 0.12, y: y + 0.08, w: 2.86, h: h - 0.16, fontFace: F, margin: 0,
        valign: "top", lineSpacingMultiple: 1.08 });
    });
    // conclusion
    s.addShape(pres.ShapeType.roundRect, { x: M, y: 5.18, w: 3.10, h: 0.52,
      fill: { color: NAVYBG }, line: { color: NAVY, width: 1 }, rectRadius: 0.05 });
    s.addText(S.painC, { x: M + 0.12, y: 5.18, w: 2.86, h: 0.52, fontFace: F,
      fontSize: 8.5, bold: true, color: NAVY, margin: 0, valign: "middle",
      lineSpacingMultiple: 1.08 });

    // ---- right zone: solution ----
    s.addShape(pres.ShapeType.ellipse, { x: RX, y: 1.19, w: 0.30, h: 0.30,
      fill: { color: TEAL }, line: { type: "none" } });
    s.addImage({ data: icons.loop, x: RX + 0.07, y: 1.26, w: 0.16, h: 0.16 });
    s.addText([
      { text: S.solH + "   ", options: { fontSize: 14, bold: true, color: NAVY } },
      { text: S.solT, options: { fontSize: 9.5, color: MUTED } },
    ], { x: RX + 0.40, y: 1.16, w: RW - 0.40, h: 0.36, fontFace: F, margin: 0,
      valign: "middle" });

    // hub box
    s.addShape(pres.ShapeType.roundRect, { x: RX, y: 1.56, w: RW, h: 1.14,
      fill: { color: NAVYBG }, line: { color: NAVY, width: 1.25 }, rectRadius: 0.06 });
    s.addText(S.hubTitle, { x: RX + 0.14, y: 1.60, w: RW - 0.28, h: 0.26,
      fontFace: F, fontSize: S.hubTitleSize, bold: true, color: NAVY, margin: 0,
      valign: "middle" });
    // engine A
    s.addShape(pres.ShapeType.roundRect, { x: RX + 0.14, y: 1.90, w: 4.55, h: 0.72,
      fill: { color: TEALBG }, line: { color: TEALD, width: 1 }, rectRadius: 0.04 });
    s.addText([
      { text: S.engA.tag + "  ", options: { fontSize: 9, bold: true, color: TEALD } },
      { text: S.engA.sub, options: { fontSize: 8.5, bold: true, color: AMBER, breakLine: true, paraSpaceAfter: 2 } },
      { text: S.engA.body, options: { fontSize: S.engineBodySize, color: INK } },
    ], { x: RX + 0.24, y: 1.94, w: 4.35, h: 0.64, fontFace: F, margin: 0,
      valign: "top", lineSpacingMultiple: 1.05 });
    // engine B
    s.addShape(pres.ShapeType.roundRect, { x: RX + 4.79, y: 1.90, w: 4.50, h: 0.72,
      fill: { color: PURBG }, line: { color: PUR, width: 1 }, rectRadius: 0.04 });
    s.addText([
      { text: S.engB.tag + "  ", options: { fontSize: 9, bold: true, color: PUR } },
      { text: S.engB.sub, options: { fontSize: 8.5, bold: true, color: AMBER, breakLine: true, paraSpaceAfter: 2 } },
      { text: S.engB.body, options: { fontSize: S.engineBodySize, color: INK } },
    ], { x: RX + 4.89, y: 1.94, w: 4.30, h: 0.64, fontFace: F, margin: 0,
      valign: "top", lineSpacingMultiple: 1.05 });

    // vertical arrows hub <-> loop row
    s.addShape(pres.ShapeType.line, { x: 4.05, y: 2.70, w: 0, h: 0.54,
      line: { color: TEAL, width: 2.25, endArrowType: "triangle" } });
    s.addShape(pres.ShapeType.line, { x: 12.55, y: 3.24, w: 0, h: -0.54,
      line: { color: AMBER, width: 2.25, endArrowType: "triangle" } });
    // release label (step 4)
    s.addShape(pres.ShapeType.ellipse, { x: 4.24, y: 2.85, w: 0.22, h: 0.22,
      fill: { color: TEAL }, line: { type: "none" } });
    s.addText(S.relN, { x: 4.24, y: 2.85, w: 0.22, h: 0.22, fontFace: F,
      fontSize: 10, bold: true, color: WHITE, align: "center", valign: "middle",
      margin: 0 });
    s.addText(S.relLabel, { x: 4.54, y: 2.74, w: 5.75, h: 0.44, fontFace: F,
      fontSize: 8, bold: true, color: TEALD, margin: 0, valign: "middle",
      lineSpacingMultiple: 1.05 });
    s.addText(S.promoteLabel, { x: 10.34, y: 2.80, w: 2.08, h: 0.32, fontFace: F,
      fontSize: 8.5, bold: true, color: AMBER, align: "right", valign: "middle",
      margin: 0 });

    // loop row (3 boxes)
    const LB = [{ x: 3.58 }, { x: 6.82 }, { x: 10.06 }];
    S.loops.forEach((b, i) => {
      const x = LB[i].x, y = 3.24, w = 2.95, h = 0.92;
      s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: WHITE },
        line: { color: NAVY, width: 1 }, rectRadius: 0.05 });
      s.addShape(pres.ShapeType.ellipse, { x: x + 0.10, y: y + 0.09, w: 0.24, h: 0.24,
        fill: { color: NAVY }, line: { type: "none" } });
      s.addText(b.n, { x: x + 0.10, y: y + 0.09, w: 0.24, h: 0.24, fontFace: F,
        fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle",
        margin: 0 });
      s.addText(b.t, { x: x + 0.42, y: y + 0.08, w: w - 0.52, h: 0.26, fontFace: F,
        fontSize: 10, bold: true, color: NAVY, margin: 0, valign: "middle" });
      s.addText(b.d, { x: x + 0.12, y: y + 0.36, w: w - 0.24, h: h - 0.44,
        fontFace: F, fontSize: 8, color: INK, margin: 0, valign: "top",
        lineSpacingMultiple: 1.06 });
      if (i < 2) {
        s.addShape(pres.ShapeType.line, { x: x + w, y: y + h / 2, w: 0.29, h: 0,
          line: { color: NAVY, width: 2, endArrowType: "triangle" } });
      }
    });

    // ---- innovations ----
    s.addShape(pres.ShapeType.ellipse, { x: RX, y: 4.29, w: 0.24, h: 0.24,
      fill: { color: AMBER }, line: { type: "none" } });
    s.addImage({ data: icons.bulb, x: RX + 0.055, y: 4.345, w: 0.13, h: 0.13 });
    s.addText(S.innoH, { x: RX + 0.34, y: 4.26, w: RW - 0.34, h: 0.30, fontFace: F,
      fontSize: 12.5, bold: true, color: NAVY, margin: 0, valign: "middle" });
    const iw = (RW - 0.42) / 4;
    S.innos.forEach((c, i) => {
      const x = RX + i * (iw + 0.14), y = 4.60, h = 1.02;
      s.addShape(pres.ShapeType.roundRect, { x, y, w: iw, h, fill: { color: WHITE },
        line: { color: LINE, width: 1 }, rectRadius: 0.05 });
      s.addText([
        { text: "0" + (i + 1) + "  ", options: { fontSize: 11, bold: true, color: TEAL } },
        { text: c.t, options: { fontSize: 9.5, bold: true, color: INK, breakLine: true, paraSpaceAfter: 3 } },
        { text: c.d, options: { fontSize: 8, color: MUTED } },
      ], { x: x + 0.10, y: y + 0.07, w: iw - 0.20, h: h - 0.14, fontFace: F,
        margin: 0, valign: "top", lineSpacingMultiple: 1.06 });
    });

    // ---- bottom strip: expected impact ----
    s.addShape(pres.ShapeType.ellipse, { x: M, y: 5.79, w: 0.24, h: 0.24,
      fill: { color: NAVY }, line: { type: "none" } });
    s.addImage({ data: icons.chart, x: M + 0.055, y: 5.845, w: 0.13, h: 0.13 });
    s.addText([
      { text: S.fxH + "   ", options: { fontSize: 12.5, bold: true, color: NAVY } },
      { text: S.fxT, options: { fontSize: 9, color: MUTED } },
    ], { x: M + 0.34, y: 5.76, w: 12.0, h: 0.30, fontFace: F, margin: 0,
      valign: "middle" });
    const tw = (W - 2 * M - 0.42) / 4;
    S.fx.forEach((t, i) => {
      const x = M + i * (tw + 0.14), y = 6.10, h = 0.76;
      s.addShape(pres.ShapeType.roundRect, { x, y, w: tw, h, fill: { color: PANEL },
        line: { color: LINE, width: 0.75 }, rectRadius: 0.05 });
      s.addText([
        { text: t.t, options: { fontSize: 11, bold: true, color: TEALD, breakLine: true, paraSpaceAfter: 2 } },
        { text: t.d, options: { fontSize: 8, color: INK } },
      ], { x: x + 0.12, y: y + 0.06, w: tw - 0.24, h: h - 0.12, fontFace: F,
        margin: 0, valign: "top", lineSpacingMultiple: 1.06 });
    });

    // facts bar
    s.addShape(pres.ShapeType.roundRect, { x: M, y: 6.98, w: W - 2 * M, h: 0.34,
      fill: { color: NAVY }, line: { type: "none" }, rectRadius: 0.05 });
    s.addShape(pres.ShapeType.roundRect, { x: M + 0.08, y: 7.035, w: 0.92, h: 0.25,
      fill: { color: TEAL }, line: { type: "none" }, rectRadius: 0.05 });
    s.addText(S.factsLead, { x: M + 0.08, y: 7.035, w: 0.92, h: 0.25, fontFace: F,
      fontSize: 8.5, bold: true, color: WHITE, align: "center", valign: "middle",
      margin: 0 });
    s.addText(S.facts, { x: M + 1.12, y: 6.98, w: W - 2 * M - 1.24, h: 0.34,
      fontFace: F, fontSize: S.factsSize, color: WHITE, margin: 0, valign: "middle" });
  }

  render(CN);
  render(EN);

  const out = process.argv[2] || "skill_hub_proposal_onepager.pptx";
  await pres.writeFile({ fileName: out });
  console.log("written:", out);
}

main().catch((e) => { console.error(e); process.exit(1); });
