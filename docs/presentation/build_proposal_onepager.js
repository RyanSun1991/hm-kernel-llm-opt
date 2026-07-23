/* Team Skill Hub — one-page proposal slide, restyled to match the shared
 * proposal template: red title, 2x2 white panels with embedded black headers,
 * black body + bold lead-ins + red emphasis, teal/orange engineering diagram. */
const pptxgen = require("pptxgenjs");

const RED = "C00000", BLACK = "1A1A1A", GRAY = "595959", MID = "404040";
const WHITE = "FFFFFF", LIGHT = "F2F2F2";
const TEALBG = "D9EEEC", TEALBG2 = "CFEDEA", TEALBD = "2E9E94";
const ORGBG = "FBE5D6", ORGBD = "ED7D31";
const F = "Microsoft YaHei";

const pres = new pptxgen();
pres.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pres.layout = "W";
const s = pres.addSlide();
s.background = { color: WHITE };

// helpers -------------------------------------------------------------------
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
function xmark(cx, cy, r, w) {
  s.addShape(pres.ShapeType.line, { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r,
    line: { color: RED, width: w } });
  s.addShape(pres.ShapeType.line, { x: cx - r, y: cy + r, w: 2 * r, h: -2 * r,
    line: { color: RED, width: w } });
}

// title ---------------------------------------------------------------------
s.addText([
  { text: "Skill Hub：", options: {} },
  { text: "鸿蒙内核智能优化的团队级自进化经验中枢", options: {} },
  { text: "（提案人 00XXXXXX）", options: {} },
], { x: 0.30, y: 0.12, w: 12.73, h: 0.52, fontFace: F, fontSize: 21,
  bold: true, color: RED, margin: 0, valign: "middle" });

// panels --------------------------------------------------------------------
panel(0.30, 0.92, 5.55, 3.70, "问题背景", 2.0);
panel(5.99, 0.92, 7.03, 3.70, "创新方案", 2.0);
panel(0.30, 4.80, 5.55, 2.48, "现有方案不足", 2.6);
panel(5.99, 4.80, 7.03, 2.48, "收益和商业价值", 2.9);

// ---- P1 问题背景 -----------------------------------------------------------
s.addText([
  { text: "● ", options: { bold: true } },
  { text: "Agent 驱动的内核优化（内存底噪 · 指令数 · 功耗）规模化铺开", options: { bold: true } },
  { text: "——真瓶颈不是「单次优化」，而是", options: {} },
  { text: "「经验复利」", options: { bold: true, color: RED, breakLine: true } },
  { text: "● ", options: { bold: true, paraSpaceBefore: 4 } },
  { text: "经验孤岛：", options: { bold: true } },
  { text: "哪个招式有效、哪里有坑、热函数怎么改、哪种验证不可信——只存在个人本地，无法跨人 / 跨项目 / 跨时间复用", options: { breakLine: true } },
  { text: "● ", options: { bold: true, paraSpaceBefore: 4 } },
  { text: "重复踩坑 · 无法传承：", options: { bold: true } },
  { text: "团队重复探索同一方向；专家经验带不走，新人与新 Agent 从零开始，", options: {} },
  { text: "优化边际效率不升反降", options: { color: RED, bold: true } },
], { x: 0.48, y: 1.06, w: 5.19, h: 1.42, fontFace: F, fontSize: 10,
  color: BLACK, margin: 0, valign: "top", lineSpacingMultiple: 1.12 });

// mini diagram: silos
const mems = ["工程师 A", "工程师 B", "工程师 C"];
mems.forEach((m, i) => {
  const x = 0.48 + i * (1.55 + 0.27);
  s.addShape(pres.ShapeType.rect, { x, y: 2.62, w: 1.55, h: 0.52,
    fill: { color: LIGHT }, line: { color: GRAY, width: 1 } });
  s.addText([
    { text: m, options: { bold: true, breakLine: true } },
    { text: "本地经验", options: { fontSize: 8 } },
  ], { x, y: 2.62, w: 1.55, h: 0.52, fontFace: F, fontSize: 9, color: BLACK,
    align: "center", valign: "middle", margin: 0, lineSpacingMultiple: 1.0 });
});
// dashed connectors from each silo down to the (missing) shared pool
[{ x: 1.255, w: 0.805 }, { x: 3.075, w: -0.055 }, { x: 4.895, w: -0.915 }]
  .forEach((c) => s.addShape(pres.ShapeType.line, { x: c.x, y: 3.14, w: c.w,
    h: 0.30, line: { color: GRAY, width: 1, dashType: "dash" } }));
xmark(2.165, 2.88, 0.07, 1.75);
xmark(3.985, 2.88, 0.07, 1.75);
s.addShape(pres.ShapeType.rect, { x: 1.42, y: 3.44, w: 3.20, h: 0.48,
  fill: { color: WHITE }, line: { color: GRAY, width: 1, dashType: "dash" } });
s.addText("团队共享经验池（缺失）", { x: 1.42, y: 3.44, w: 3.20, h: 0.48,
  fontFace: F, fontSize: 9.5, color: GRAY, align: "center", valign: "middle",
  margin: 0 });
s.addShape(pres.ShapeType.line, { x: 1.42, y: 3.44, w: 3.20, h: 0.48,
  line: { color: RED, width: 1.5 } });
s.addShape(pres.ShapeType.line, { x: 1.42, y: 3.92, w: 3.20, h: -0.48,
  line: { color: RED, width: 1.5 } });
s.addText("经验随人员流动而流失；新人 / 新 Agent 每次从零开始", {
  x: 0.48, y: 4.06, w: 5.19, h: 0.26, fontFace: F, fontSize: 9.5, bold: true,
  color: RED, align: "center", valign: "middle", margin: 0 });

// ---- P3 现有方案不足 -------------------------------------------------------
s.addText([
  { text: "☐ ", options: {} },
  { text: "业界 Agent 记忆方案（mem0 / Zep / memU）只解决「存与取」，", options: {} },
  { text: "没有团队级治理", options: { bold: true, breakLine: true } },
  { text: "☐ ", options: { paraSpaceBefore: 3 } },
  { text: "风险① 知识漂移：", options: { bold: true, color: RED } },
  { text: "经验自相矛盾、随时间失效，越积越乱", options: { breakLine: true } },
  { text: "☐ ", options: { paraSpaceBefore: 3 } },
  { text: "风险② 反馈自增强：", options: { bold: true, color: RED } },
  { text: "自积累让均值变好，个别场景被悄悄做坏", options: { breakLine: true } },
  { text: "☐ ", options: { paraSpaceBefore: 3 } },
  { text: "简单 git 共享也不行：", options: { bold: true } },
  { text: "行级合并让知识自相矛盾、技能被某人一周的坏经验覆盖", options: { breakLine: true } },
  { text: "☐ ", options: { paraSpaceBefore: 3 } },
  { text: "无质量门 · 无审计 · 无版本——", options: {} },
  { text: "规模越大，污染风险越大", options: { bold: true, breakLine: true } },
  { text: "∴ 需要：带质量治理、可自进化的团队级经验基础设施", options: { bold: true, color: RED, paraSpaceBefore: 7 } },
], { x: 0.48, y: 5.00, w: 5.19, h: 2.20, fontFace: F, fontSize: 10,
  color: BLACK, margin: 0, valign: "top", lineSpacingMultiple: 1.18 });

// ---- P2 创新方案 -----------------------------------------------------------
s.addText([
  { text: "核心思想：", options: { bold: true, color: RED } },
  { text: "把团队优化经验治理成「私有 npm 包」——全自动闭环、越用越准", options: { bold: true, color: RED } },
], { x: 6.17, y: 1.04, w: 6.67, h: 0.26, fontFace: F, fontSize: 11,
  margin: 0, valign: "middle" });

// hub box
s.addShape(pres.ShapeType.rect, { x: 6.17, y: 1.38, w: 6.67, h: 0.88,
  fill: { color: TEALBG }, line: { color: TEALBD, width: 1.25 } });
s.addText([
  { text: "hm-skill-hub 团队中央经验仓：", options: { bold: true } },
  { text: "semver 发布 + lockfile 钉版 · 可回滚 · 全程可审计", options: {} },
], { x: 6.31, y: 1.42, w: 6.4, h: 0.20, fontFace: F, fontSize: 9.5,
  color: BLACK, margin: 0, valign: "middle" });
s.addShape(pres.ShapeType.rect, { x: 6.31, y: 1.66, w: 3.16, h: 0.52,
  fill: { color: TEALBG2 }, line: { color: TEALBD, width: 1 } });
s.addText([
  { text: "Knowledge 知识 · 引擎A 治理型合并", options: { bold: true, breakLine: true } },
  { text: "只追加 + 七路关系分类 · 双时态墓碑不物删", options: {} },
], { x: 6.40, y: 1.66, w: 3.00, h: 0.52, fontFace: F, fontSize: 8,
  color: BLACK, margin: 0, valign: "middle", lineSpacingMultiple: 1.05 });
s.addShape(pres.ShapeType.rect, { x: 9.57, y: 1.66, w: 3.13, h: 0.52,
  fill: { color: ORGBG }, line: { color: ORGBD, width: 1 } });
s.addText([
  { text: "Skills 技能 · 引擎B 竞争式进化", options: { bold: true, breakLine: true } },
  { text: "SkillOpt 有界编辑 · eval 严格变好 · Pareto 防塌缩", options: {} },
], { x: 9.66, y: 1.66, w: 3.00, h: 0.52, fontFace: F, fontSize: 8,
  color: BLACK, margin: 0, valign: "middle", lineSpacingMultiple: 1.05 });

// arrows + step 4 label
s.addShape(pres.ShapeType.line, { x: 6.55, y: 2.26, w: 0, h: 0.46,
  line: { color: TEALBD, width: 2, endArrowType: "triangle" } });
s.addText([
  { text: "④ 发布回灌：", options: { bold: true, color: RED } },
  { text: "nightly 七步 → semver + scorecard → broadcast 自动钉版", options: { color: BLACK } },
], { x: 6.75, y: 2.32, w: 4.55, h: 0.30, fontFace: F, fontSize: 8.5,
  margin: 0, valign: "middle" });
s.addShape(pres.ShapeType.line, { x: 12.72, y: 2.72, w: 0, h: -0.46,
  line: { color: ORGBD, width: 2, endArrowType: "triangle" } });
s.addText("过三道门 → 入库", { x: 11.30, y: 2.32, w: 1.34, h: 0.30,
  fontFace: F, fontSize: 8.5, bold: true, color: BLACK, align: "right",
  valign: "middle", margin: 0 });

// loop row
const LOOPS = [
  { t: "① 消费 · 自动挂载", d: "MCP · 6 接入点 · 静默降级" },
  { t: "② 收口 · 自动蒸馏", d: "规则 + LLM 双段抽取" },
  { t: "③ 门控 · 策展入库", d: "CI 五道门 + 双人评审" },
];
LOOPS.forEach((b, i) => {
  const x = 6.17 + i * (2.06 + 0.23);
  s.addShape(pres.ShapeType.rect, { x, y: 2.72, w: 2.06, h: 0.58,
    fill: { color: WHITE }, line: { color: MID, width: 1 } });
  s.addText([
    { text: b.t, options: { bold: true, breakLine: true } },
    { text: b.d, options: { fontSize: 7.5, color: MID } },
  ], { x: x + 0.08, y: 2.72, w: 2.06 - 0.16, h: 0.58, fontFace: F,
    fontSize: 8.5, color: BLACK, margin: 0, valign: "middle",
    lineSpacingMultiple: 1.05 });
  if (i < 2) {
    s.addShape(pres.ShapeType.line, { x: x + 2.06, y: 3.01, w: 0.23, h: 0,
      line: { color: MID, width: 1.75, endArrowType: "triangle" } });
  }
});

// 创新点 bullets
s.addText([
  { text: "创新点1 双资产双引擎：", options: { bold: true, color: RED } },
  { text: "知识与技能分治、各配治理引擎，根除「漂移」与「坏经验覆盖」——业界记忆方案缺失的治理层", options: { breakLine: true } },
  { text: "创新点2 eval-gate 反喂安全：", options: { bold: true, color: RED, paraSpaceBefore: 3 } },
  { text: "技能改动须留出套件「严格变好且零回归」才入库，根治「自增强跑偏」", options: { breakLine: true } },
  { text: "创新点3 全自动闭环：", options: { bold: true, color: RED, paraSpaceBefore: 3 } },
  { text: "蒸馏 → 门控 → 发布 → 回灌全自动，≥2 独立实例自动毕业为技能，人只握审批权；MCP 接入，故障静默降级不阻塞", options: {} },
], { x: 6.17, y: 3.46, w: 6.67, h: 1.12, fontFace: F, fontSize: 9.5,
  color: BLACK, margin: 0, valign: "top", lineSpacingMultiple: 1.12 });

// ---- P4 收益和商业价值 -----------------------------------------------------
s.addText([
  { text: "新人 / 新 Agent 开箱即带全队经验；经验复利，吞吐与命中率随使用持续提升", options: { bold: true, color: RED } },
], { x: 6.17, y: 4.96, w: 6.67, h: 0.26, fontFace: F, fontSize: 10.5,
  margin: 0, valign: "middle" });
s.addText([
  { text: "☐ ", options: {} },
  { text: "提效：", options: { bold: true } },
  { text: "重复探索与重复踩坑显著减少；专家经验沉淀为质量门保护的团队资产，不随人员流失", options: { breakLine: true } },
  { text: "☐ ", options: { paraSpaceBefore: 3 } },
  { text: "通用：", options: { bold: true } },
  { text: "内存底噪 / 指令数 / 功耗 / 编译器后端等任意 Agent 优化场景即插即用的自进化记忆底座", options: { breakLine: true } },
  { text: "☐ ", options: { paraSpaceBefore: 3 } },
  { text: "落地验证：", options: { bold: true, color: RED } },
  { text: "全链路已实现、离线可复现——154 项测试 · 5 道 CI 门 · 3 个 MCP 工具 · 检索 recall@5=", options: {} },
  { text: "1.0", options: { bold: true, color: RED } },
  { text: " · 七路分类 ", options: {} },
  { text: "48/48", options: { bold: true, color: RED } },
  { text: " · 优化门 ", options: {} },
  { text: "0.67→1.00", options: { bold: true, color: RED, breakLine: true } },
  { text: "☐ ", options: { paraSpaceBefore: 3 } },
  { text: "前沿：", options: { bold: true } },
  { text: "直面 self-evolving agent / 长期记忆研究热点——安全积累 · 矛盾治理 · 全程审计", options: {} },
], { x: 6.17, y: 5.30, w: 6.67, h: 1.88, fontFace: F, fontSize: 10,
  color: BLACK, margin: 0, valign: "top", lineSpacingMultiple: 1.12 });

const out = process.argv[2] || "skill_hub_proposal_onepager.pptx";
pres.writeFile({ fileName: out }).then(() => console.log("written:", out));
