/* Team Skill Hub — one-page proposal slide v3.
 * Template-matched (red title, 2x2 bordered panels, embedded headers) with
 * reference-grade density: layered architecture diagram with real component
 * names, capability comparison table, and two native data charts. */
const pptxgen = require("pptxgenjs");

const RED = "C00000", BLACK = "1A1A1A", GRAY = "595959", MID = "404040";
const WHITE = "FFFFFF", LIGHT = "F2F2F2", SOFT = "808080";
const TEALBG = "D9EEEC", TEALBG2 = "CFEDEA", TEALBD = "2E9E94", TEALD = "0B7A70";
const ORGBG = "FBE5D6", ORGBD = "ED7D31";
const F = "Microsoft YaHei";

const pres = new pptxgen();
pres.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pres.layout = "W";
const s = pres.addSlide();
s.background = { color: WHITE };

function panel(x, y, w, h, title, titleW) {
  s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: { color: WHITE },
    line: { color: GRAY, width: 1.25 } });
  const tx = x + (w - titleW) / 2;
  s.addShape(pres.ShapeType.rect, { x: tx, y: y - 0.17, w: titleW, h: 0.34,
    fill: { color: WHITE }, line: { type: "none" } });
  s.addText(title, { x: tx, y: y - 0.17, w: titleW, h: 0.34, fontFace: F,
    fontSize: 16, bold: true, color: BLACK, align: "center", valign: "middle",
    margin: 0 });
}
function box(x, y, w, h, fill, bd, bw) {
  s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: { color: fill },
    line: { color: bd, width: bw || 1 } });
}
function arrow(x, y, w, h, color, wd, dash) {
  const ln = { color, width: wd || 1.5, endArrowType: "triangle" };
  if (dash) ln.dashType = "dash";
  s.addShape(pres.ShapeType.line, { x, y, w, h, line: ln });
}
function dline(x, y, w, h) {
  s.addShape(pres.ShapeType.line, { x, y, w, h,
    line: { color: SOFT, width: 1, dashType: "dash" } });
}

// ---- title ----------------------------------------------------------------
s.addText([
  { text: "Skill Hub：", options: {} },
  { text: "鸿蒙内核智能优化的团队级自进化经验中枢", options: {} },
  { text: "（提案人 00XXXXXX）", options: {} },
], { x: 0.30, y: 0.12, w: 12.73, h: 0.52, fontFace: F, fontSize: 21,
  bold: true, color: RED, margin: 0, valign: "middle" });

panel(0.30, 0.92, 5.55, 3.70, "问题背景", 2.0);
panel(5.99, 0.92, 7.03, 3.70, "创新方案", 2.0);
panel(0.30, 4.80, 5.55, 2.48, "现有方案不足", 2.6);
panel(5.99, 4.80, 7.03, 2.48, "收益和商业价值", 2.9);

// ============================== P1 问题背景 ================================
s.addText([
  { text: "● ", options: { bold: true } },
  { text: "Agent 驱动的内核优化（内存底噪 · 指令数 · 功耗）规模化铺开", options: { bold: true } },
  { text: "——真瓶颈不是「单次优化」，而是", options: {} },
  { text: "「经验能否复利积累」", options: { bold: true, color: RED, breakLine: true } },
  { text: "● ", options: { bold: true, paraSpaceBefore: 3 } },
  { text: "哪个招式有效、哪里有坑、热函数怎么改、哪种验证不可信——只存在个人本地；团队重复探索、专家经验带不走，", options: {} },
  { text: "优化边际效率不升反降", options: { bold: true, color: RED } },
], { x: 0.48, y: 1.04, w: 5.19, h: 0.98, fontFace: F, fontSize: 10,
  color: BLACK, margin: 0, valign: "top", lineSpacingMultiple: 1.1 });

// As-Is flow: engineers -> pipeline -> local memory -> broken reuse
const CHIPX = [0.48, 2.30, 4.12];
CHIPX.forEach((x, i) => {
  box(x, 2.08, 1.55, 0.28, LIGHT, SOFT, 1);
  s.addText("工程师 " + "ABC"[i] + "（各自为战）", { x, y: 2.08, w: 1.55, h: 0.28,
    fontFace: F, fontSize: 8, bold: true, color: BLACK, align: "center",
    valign: "middle", margin: 0 });
  arrow(x + 0.775, 2.36, 0, 0.13, SOFT, 1.25);
});
box(0.48, 2.50, 5.19, 0.40, WHITE, MID, 1);
s.addText([
  { text: "Agent 优化流水线（.opencode 硬门禁）：", options: { bold: true } },
  { text: "research → 计划评审 → 实现 → 代码评审 → 测试 A/B → 决策", options: {} },
], { x: 0.56, y: 2.50, w: 5.03, h: 0.40, fontFace: F, fontSize: 8,
  color: BLACK, margin: 0, valign: "middle", lineSpacingMultiple: 1.05 });
arrow(3.075, 2.90, 0, 0.12, SOFT, 1.25);
box(0.48, 3.03, 5.19, 0.42, "FFF8F0", ORGBD, 1);
s.addText([
  { text: "本地经验 .opencode/memory/：", options: { bold: true } },
  { text: "idea_ledger（L001 landed −0.8%）· targets 结构事实 · global_lessons · bad_plans", options: {} },
], { x: 0.56, y: 3.03, w: 5.03, h: 0.42, fontFace: F, fontSize: 8,
  color: BLACK, margin: 0, valign: "middle", lineSpacingMultiple: 1.05 });
CHIPX.forEach((x, i) => {
  dline(3.075 + (i - 1) * 0.02, 3.45, x + 0.775 - 3.075, 0.16);
  box(x, 3.62, 1.55, 0.30, WHITE, SOFT, 1);
  s.addText([
    { text: ["队友复用 ", "新人接手 ", "新 Agent "][i], options: {} },
    { text: "✗", options: { bold: true, color: RED } },
  ], { x, y: 3.62, w: 1.55, h: 0.30, fontFace: F, fontSize: 8.5, color: BLACK,
    align: "center", valign: "middle", margin: 0 });
});
s.addText("经验止步个人目录——无汇聚 · 无治理 · 随人流失，新人与新 Agent 永远从零开始", {
  x: 0.48, y: 3.99, w: 5.19, h: 0.22, fontFace: F, fontSize: 9.5, bold: true,
  color: RED, align: "center", valign: "middle", margin: 0 });
s.addText("例：shrink_node「hoist sc->priority」bench 实测 −0.8%——这类可复用经验此前只躺在个人目录", {
  x: 0.48, y: 4.26, w: 5.19, h: 0.20, fontFace: F, fontSize: 7.5, color: GRAY,
  align: "center", valign: "middle", margin: 0 });

// ============================== P3 现有方案不足 =============================
s.addText([
  { text: "业界记忆方案只解决「个体的存与取」；", options: {} },
  { text: "知识漂移 · 反馈自增强", options: { bold: true, color: RED } },
  { text: " 两大致命风险与团队级复用均无人治理：", options: {} },
], { x: 0.48, y: 4.94, w: 5.19, h: 0.40, fontFace: F, fontSize: 9.5,
  color: BLACK, margin: 0, valign: "top", lineSpacingMultiple: 1.1 });

const th = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: GRAY }, align: "center", fontSize: 7.5 } });
const ok = (t, hl) => ({ text: t, options: { color: TEALD, bold: true, align: "center", fontSize: 7.5, fill: { color: hl ? TEALBG : WHITE } } });
const no = (t) => ({ text: t, options: { color: RED, align: "center", fontSize: 7.5, fill: { color: WHITE } } });
const pt = (t) => ({ text: t, options: { color: GRAY, align: "center", fontSize: 7.5, fill: { color: WHITE } } });
const nm = (t, hl) => ({ text: t, options: { bold: true, align: "left", fontSize: 7.5, color: hl ? TEALD : BLACK, fill: { color: hl ? TEALBG : WHITE } } });
s.addTable([
  [th("方案"), th("存取检索"), th("冲突治理"), th("质量门"), th("版本化发布"), th("团队策展")],
  [nm("mem0 v3 OSS"), ok("✓"), no("✗ ADD-only"), no("✗"), no("✗"), no("✗")],
  [nm("Zep / Graphiti"), ok("✓"), pt("△ 双时态"), no("✗"), no("✗"), no("✗")],
  [nm("memU / EverOS"), ok("✓ 分层"), no("✗"), no("✗"), no("✗"), no("✗")],
  [nm("git 直接共享"), pt("△"), no("✗ 行级冲突"), no("✗"), pt("△"), no("✗")],
  [nm("Skill Hub 本方案", 1), ok("✓ 混合检索", 1), ok("✓ 七路+双时态", 1), ok("✓ 3门+eval", 1), ok("✓ semver", 1), ok("✓ 双评审", 1)],
], { x: 0.44, y: 5.38, w: 5.27, colW: [1.18, 0.82, 1.00, 0.78, 0.87, 0.62],
  rowH: 0.235, border: { pt: 0.75, color: "A6A6A6" }, fontFace: F,
  valign: "middle", margin: 0.03 });
s.addText([
  { text: "∴ 团队级治理层（门控 · 策展 · 版本化）为本方案独有——差异化护城河", options: { bold: true, color: RED } },
], { x: 0.48, y: 6.94, w: 5.19, h: 0.24, fontFace: F, fontSize: 9.5,
  margin: 0, valign: "middle" });

// ============================== P2 创新方案 ================================
s.addText([
  { text: "核心思想：", options: { bold: true, color: RED } },
  { text: "经验如「私有 npm 包」——消费→蒸馏→门控→发布→再消费，全自动闭环、越用越准", options: { bold: true, color: RED } },
], { x: 6.15, y: 1.00, w: 6.71, h: 0.24, fontFace: F, fontSize: 10,
  margin: 0, valign: "middle" });

// -- left: layered architecture (hub lane + member lane), 7 numbered steps --
const DX = 6.15, DW = 4.55;
box(DX, 1.30, DW, 1.36, "F4FAF9", TEALBD, 1.25);
s.addText([
  { text: "hm-skill-hub 团队中央经验仓", options: { bold: true } },
  { text: "（semver 发布 · lockfile 钉版 · 可回滚 · 可审计）", options: { fontSize: 7 } },
], { x: DX + 0.12, y: 1.34, w: DW - 0.24, h: 0.18, fontFace: F, fontSize: 8,
  color: BLACK, margin: 0, valign: "middle" });
// row A: staging <- CI <- curation (flow right-to-left feeds the stores)
const RA = [
  { t: "⑥ 策展入库", d: "七路分类 · 双人评审" },
  { t: "⑤ CI 五道门", d: "lint·脱敏·去重·eval·测试" },
  { t: "④ staging 收件箱", d: "成员蒸馏包投稿 PR" },
];
RA.forEach((b, i) => {
  const x = DX + 0.12 + i * (1.36 + 0.115);
  box(x, 1.56, 1.36, 0.34, WHITE, MID, 1);
  s.addText([
    { text: b.t, options: { bold: true, breakLine: true } },
    { text: b.d, options: { fontSize: 6, color: GRAY } },
  ], { x: x + 0.03, y: 1.56, w: 1.30, h: 0.34, fontFace: F, fontSize: 6.5,
    color: BLACK, align: "center", valign: "middle", margin: 0,
    lineSpacingMultiple: 1.0 });
  if (i < 2) arrow(x + 1.36 + 0.115, 1.73, -0.115, 0, MID, 1.25);
});
// row B: the two stores (dual engines)
box(DX + 0.12, 1.96, 2.12, 0.60, TEALBG2, TEALBD, 1);
s.addText([
  { text: "knowledge/ 知识库 — 引擎A 治理合并", options: { bold: true, breakLine: true } },
  { text: "global · subsystems · targets（F/H/A/V/B/L）", options: { fontSize: 6, breakLine: true } },
  { text: "七路关系分类 · 双时态墓碑 · 永不物删", options: { fontSize: 6.5 } },
], { x: DX + 0.18, y: 1.96, w: 2.00, h: 0.60, fontFace: F, fontSize: 7,
  color: BLACK, margin: 0, valign: "middle", lineSpacingMultiple: 1.05 });
box(DX + 2.31, 1.96, 2.12, 0.60, ORGBG, ORGBD, 1);
s.addText([
  { text: "skills/ 技能库 — 引擎B 竞争进化", options: { bold: true, breakLine: true } },
  { text: "core · technique · domain + best_skill.md", options: { fontSize: 6, breakLine: true } },
  { text: "SkillOpt 有界编辑 · eval 门 · GEPA/Pareto", options: { fontSize: 6.5 } },
], { x: DX + 2.37, y: 1.96, w: 2.00, h: 0.60, fontFace: F, fontSize: 7,
  color: BLACK, margin: 0, valign: "middle", lineSpacingMultiple: 1.05 });

// publish arrow (hub -> member lane) + labels
arrow(6.50, 2.66, 0, 0.26, TEALBD, 2);
s.addText([
  { text: "⑦ 发布回灌：", options: { bold: true, color: RED } },
  { text: "nightly 七步 · GEPA 寻优 · semver · broadcast 钉版", options: { color: BLACK } },
], { x: 6.68, y: 2.66, w: 3.92, h: 0.24, fontFace: F, fontSize: 7.5,
  margin: 0, valign: "middle" });
// member lane: 3 stacked stages
const ML = [
  [{ text: "① 消费 · 自动挂载：", options: { bold: true } },
   { text: "skill-memory.lock 钉版 + resolve 注入「Hub 上下文」", options: {} }],
  [{ text: "② Agent 优化流水线：", options: { bold: true } },
   { text: "research → 计划评审 → 实现 → 代码评审 → 测试 → 决策", options: {} }],
  [{ text: "③ 收口蒸馏 sediment：", options: { bold: true } },
   { text: "规则 + LLM 双段抽取 → Tier-1 候选 → ④ 投稿 PR", options: {} }],
];
ML.forEach((runs, i) => {
  const y = 2.92 + i * 0.44;
  box(DX, y, 4.43, 0.32, WHITE, MID, 1);
  s.addText(runs, { x: DX + 0.10, y, w: 4.23, h: 0.32, fontFace: F,
    fontSize: 7.5, color: BLACK, margin: 0, valign: "middle" });
  if (i < 2) arrow(8.36, y + 0.32, 0, 0.12, SOFT, 1.25);
});
// PR submission arrow back up into staging (right edge)
arrow(10.66, 3.96, 0, -1.30, ORGBD, 2);
s.addText("MCP：resolve / sediment / status（7338）· hub 不可达静默降级、不阻塞", {
  x: DX, y: 4.36, w: DW, h: 0.18, fontFace: F, fontSize: 7, color: GRAY,
  margin: 0, valign: "middle" });

// -- right: innovation column (KSPECT-style red-headed blocks) --
const IX = 10.88, IW = 1.98;
const INNO = [
  { h: "创新点1：双资产双引擎", b: [
    { text: "知识＝七路关系分类 + 双时态墓碑，永不物删；技能＝SkillOpt + Pareto，eval 门内进化——", options: {} },
    { text: "业界记忆方案缺失的治理层", options: { bold: true } }] },
  { h: "创新点2：eval门×GEPA 进化", b: [
    { text: "技能改动须留出套件", options: {} },
    { text: "「严格变好且零回归」", options: { bold: true } },
    { text: "才入库，根治「自增强跑偏」；引入 GEPA 反思进化 + Pareto 前沿自动寻优技能文本", options: {} }] },
  { h: "创新点3：零侵入 · 好上手", b: [
    { text: "研发习惯零改动：", options: { bold: true } },
    { text: "上下文自动挂载、收口自动蒸馏；MCP 即插即用、故障静默降级；蒸馏到发布全自动闭环，人只做晋升审批", options: {} }] },
];
let iy = 1.32;
INNO.forEach((blk) => {
  s.addText(blk.h, { x: IX, y: iy, w: IW, h: 0.20, fontFace: F, fontSize: 8.5,
    bold: true, color: RED, margin: 0, valign: "middle" });
  s.addText(blk.b, { x: IX, y: iy + 0.21, w: IW, h: 0.78, fontFace: F,
    fontSize: 7.5, color: BLACK, margin: 0, valign: "top",
    lineSpacingMultiple: 1.08 });
  iy += 1.06;
});

// ============================== P4 收益和商业价值 ===========================
s.addText([
  { text: "新人 / 新 Agent 开箱即带全队经验；经验跨人跨迭代复利，吞吐与命中率随使用持续提升", options: { bold: true, color: RED } },
], { x: 6.15, y: 4.92, w: 6.71, h: 0.26, fontFace: F, fontSize: 10,
  margin: 0, valign: "middle" });
s.addText([
  { text: "☐ ", options: {} },
  { text: "提效：", options: { bold: true } },
  { text: "告别重复探索与重复踩坑；专家经验成为质量门保护的团队资产，不随人员流失", options: { breakLine: true } },
  { text: "☐ ", options: { paraSpaceBefore: 3 } },
  { text: "易用：", options: { bold: true } },
  { text: "研发习惯零改动、MCP 即插即用；新人与新 Agent 冷启动预期", options: {} },
  { text: "从周级降到天级", options: { bold: true, color: RED, breakLine: true } },
  { text: "☐ ", options: { paraSpaceBefore: 3 } },
  { text: "通用：", options: { bold: true } },
  { text: "底噪 / 指令数 / 功耗 / 编译器后端任意 Agent 优化场景即插即用", options: { breakLine: true } },
  { text: "☐ ", options: { paraSpaceBefore: 3 } },
  { text: "前沿：", options: { bold: true } },
  { text: "self-evolving agent 长期记忆——安全积累 · 矛盾治理 · 全程审计", options: {} },
], { x: 6.15, y: 5.26, w: 3.30, h: 1.92, fontFace: F, fontSize: 9.5,
  color: BLACK, margin: 0, valign: "top", lineSpacingMultiple: 1.12 });

// charts: expected-benefit visuals (labelled 示意/目标, not measured results)
s.addText([
  { text: "经验资产复利（示意）", options: { bold: true, breakLine: true } },
  { text: "有 Hub 复利 · 无 Hub 随人清零", options: { color: GRAY, fontSize: 7 } },
], { x: 9.58, y: 5.24, w: 1.84, h: 0.32, fontFace: F, fontSize: 7.5,
  color: BLACK, margin: 0, valign: "top", lineSpacingMultiple: 1.0 });
s.addChart(pres.ChartType.line, [
  { name: "无 Hub", labels: ["1", "2", "3", "4", "5", "6"], values: [10, 12, 9, 13, 10, 12] },
  { name: "有 Hub", labels: ["1", "2", "3", "4", "5", "6"], values: [10, 22, 36, 52, 70, 90] },
], { x: 9.56, y: 5.58, w: 1.84, h: 1.30, lineSize: 1.75, lineSmooth: true,
  lineDataSymbol: "none", chartColors: ["BFBFBF", "C00000"],
  catAxisLabelFontSize: 6, catAxisLabelColor: MID, catAxisLabelFontFace: F,
  valAxisHidden: true, valAxisMinVal: 0,
  valGridLine: { style: "none" }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "b", legendFontSize: 6, legendFontFace: F,
  showValue: false, showTitle: false });
s.addText([
  { text: "预期收益目标", options: { bold: true, breakLine: true } },
  { text: "相对现状改善幅度（%）", options: { color: GRAY, fontSize: 7 } },
], { x: 11.48, y: 5.24, w: 1.40, h: 0.32, fontFace: F, fontSize: 7.5,
  color: BLACK, margin: 0, valign: "top", lineSpacingMultiple: 1.0 });
s.addChart(pres.ChartType.bar, [
  { name: "改善幅度", labels: ["重复探索", "冷启动", "命中率"], values: [50, 70, 30] },
], { x: 11.46, y: 5.58, w: 1.42, h: 1.30, barDir: "col", barGapWidthPct: 60,
  chartColors: ["2E9E94"], showValue: true,
  dataLabelPosition: "outEnd", dataLabelFontSize: 6.5, dataLabelColor: MID,
  dataLabelFormatCode: '0"%"', dataLabelFontFace: F,
  catAxisLabelFontSize: 6, catAxisLabelColor: MID, catAxisLabelFontFace: F,
  valAxisHidden: true, valAxisMaxVal: 85, valAxisMinVal: 0,
  valGridLine: { style: "none" }, catGridLine: { style: "none" },
  showLegend: false, showTitle: false });
s.addText([
  { text: "进展：双仓全链路已打通（154 项测试 · 5 道 CI 门 · 离线基准全绿）", options: { breakLine: true } },
  { text: "GEPA 反思进化集成中 · 真机收益数据随试点回填", options: {} },
], { x: 9.56, y: 6.90, w: 3.32, h: 0.34, fontFace: F, fontSize: 6.5, color: GRAY,
  margin: 0, valign: "top", lineSpacingMultiple: 1.1 });

const out = process.argv[2] || "skill_hub_onepager_v3.pptx";
pres.writeFile({ fileName: out }).then(() => console.log("written:", out));
