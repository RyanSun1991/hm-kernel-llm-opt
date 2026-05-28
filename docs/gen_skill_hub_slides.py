#!/usr/bin/env python3
"""Generate a presentation deck for the Team Skill Hub design.

Native, editable PowerPoint shapes (16:9). Regenerate with:
    pip install python-pptx
    python3 docs/gen_skill_hub_slides.py
Output: docs/Team_Skill_Hub_Design_Slides.pptx
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

# ---- palette ---------------------------------------------------------------
NAVY = RGBColor(0x0F, 0x1E, 0x3D)
NAVY2 = RGBColor(0x1B, 0x2E, 0x55)
INK = RGBColor(0x1F, 0x29, 0x37)
MUT = RGBColor(0x6B, 0x72, 0x80)
BLUE = RGBColor(0x25, 0x63, 0xEB)
INDIGO = RGBColor(0x4F, 0x46, 0xE5)
TEAL = RGBColor(0x0D, 0x94, 0x88)
AMBER = RGBColor(0xB4, 0x7E, 0x00)
RED = RGBColor(0xDC, 0x35, 0x45)
GREEN = RGBColor(0x16, 0xA3, 0x4A)
ORANGE = RGBColor(0xEA, 0x58, 0x0C)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
L_INDIGO = RGBColor(0xEE, 0xF2, 0xFF)
L_BLUE = RGBColor(0xE3, 0xED, 0xFF)
L_TEAL = RGBColor(0xE2, 0xF7, 0xF3)
L_AMBER = RGBColor(0xFD, 0xF4, 0xDC)
L_ORANGE = RGBColor(0xFF, 0xEC, 0xDC)
L_GREEN = RGBColor(0xDD, 0xF3, 0xE5)
L_GRAY = RGBColor(0xF1, 0xF5, 0xF9)
LINE = RGBColor(0xCB, 0xD5, 0xE1)
LIGHTTXT = RGBColor(0xC8, 0xD2, 0xE0)

FONT = "Microsoft YaHei"
EW, EH = 13.333, 7.5


def set_run(r, size, color, bold=False, font=FONT):
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    rPr = r._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", font)


def no_shadow(shape):
    spPr = shape._element.spPr
    if spPr.find(qn("a:effectLst")) is None:
        spPr.append(spPr.makeelement(qn("a:effectLst"), {}))


def settext(shape, lines, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=3):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    first = True
    for txt, size, color, bold in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.space_after = Pt(space)
        p.space_before = Pt(0)
        r = p.add_run()
        r.text = txt
        set_run(r, size, color, bold)


def box(slide, x, y, w, h, fill, line=LINE, radius=0.09, lw=1.0):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    no_shadow(s)
    try:
        s.adjustments[0] = radius
    except Exception:
        pass
    return s


def rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    no_shadow(s)
    return s


def shape(slide, kind, x, y, w, h, fill, line=None, lw=1.0):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    no_shadow(s)
    return s


def textbox(slide, x, y, w, h, lines, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    settext(tb, lines, align, anchor)
    return tb


def arrow(slide, x1, y1, x2, y2, color=BLUE, weight=2.25, dash=False):
    cn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    cn.line.color.rgb = color
    cn.line.width = Pt(weight)
    ln = cn.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    if dash:
        ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))
    return cn


def badge(slide, cx, cy, label, color=INDIGO, w=1.15, h=0.36):
    b = box(slide, cx - w / 2, cy - h / 2, w, h, color, line=None, radius=0.5)
    settext(b, [(label, 11, WHITE, True)])
    return b


def header(slide, title, sub, page):
    rect(slide, 0, 0, EW, 1.0, NAVY)
    rect(slide, 0, 1.0, EW, 0.07, TEAL)
    textbox(slide, 0.5, 0.10, 11.2, 0.55, [(title, 23, WHITE, True)], anchor=MSO_ANCHOR.MIDDLE)
    textbox(slide, 0.52, 0.60, 11.2, 0.34, [(sub, 11, LIGHTTXT, False)])
    textbox(slide, 12.2, 0.30, 0.9, 0.5, [(page, 12, LIGHTTXT, True)], align=PP_ALIGN.RIGHT)


def footer(slide, txt="Team Skill Hub · Draft v2.0 · 2026-05"):
    textbox(slide, 0.5, 7.15, 12.3, 0.3, [(txt, 9, MUT, False)])


def takeaway(slide, y, txt, fill=L_INDIGO, line=INDIGO, color=INDIGO):
    b = box(slide, 0.6, y, 12.13, 0.6, fill, line=line, radius=0.12)
    settext(b, [(txt, 12.5, color, True)])


# ---- deck ------------------------------------------------------------------
def build():
    prs = Presentation()
    prs.slide_width = Inches(EW)
    prs.slide_height = Inches(EH)
    blank = prs.slide_layouts[6]

    # S1 cover -------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    rect(s, 0, 0, EW, EH, NAVY)
    rect(s, 0, 0, 0.28, EH, TEAL)
    rect(s, 0.9, 4.30, 6.6, 0.04, TEAL)
    textbox(s, 0.9, 2.25, 11.4, 1.2, [("团队级 Skill / Memory 仓库闭环方案", 40, WHITE, True)])
    textbox(s, 0.92, 3.45, 11.4, 0.6, [("Team Skill Hub  ·  本地沉淀 → 团队技能仓 → 反哺 Pipeline → 闭环迭代", 17, TEAL, False)])
    textbox(s, 0.92, 4.45, 11.4, 0.5,
            [("可迭代  ·  可沉淀  ·  可闭环优化  ·  可复用  ·  稳定可用", 16, LIGHTTXT, False)])
    pills = ["SkillOpt", "memU", "Mem0 / Zep", "GEPA", "Voyager", "Agent Skills"]
    px = 0.9
    for p in pills:
        w = 0.45 + 0.16 * len(p)
        b = box(s, px, 5.55, w, 0.5, NAVY2, line=TEAL, radius=0.5, lw=1.0)
        settext(b, [(p, 12, WHITE, True)])
        px += w + 0.25
    textbox(s, 0.9, 6.7, 8, 0.4, [("Draft v2.0  ·  2026-05  ·  研究 + 方案设计", 11, LIGHTTXT, False)])

    # S2 closed loop (hero) -----------------------------------------------
    s = prs.slides.add_slide(blank)
    header(s, "闭环全景 · 两仓协同", "消费 → 蒸馏 → 沉淀合并 → eval 发布 → 再消费（每环都有质量门）", "02")
    by, bh = 2.55, 1.75
    A = box(s, 0.55, by, 2.75, bh, L_INDIGO, line=INDIGO, radius=0.1)
    settext(A, [("hm-skill-hub", 14, INDIGO, True), ("团队资产面 · semver", 10, MUT, False),
                ("skills/ (引擎B)", 11, INK, False), ("knowledge/ (引擎A)", 11, INK, False)])
    B = box(s, 3.85, by, 2.95, bh, L_TEAL, line=TEAL, radius=0.1)
    settext(B, [("pipeline 运行", 14, TEAL, True), ("（每成员）", 10, MUT, False),
                ("research→plan→code→", 10.5, INK, False), ("review→test→decision", 10.5, INK, False)])
    C = box(s, 7.35, by, 2.55, bh, L_GRAY, line=LINE, radius=0.1)
    settext(C, [("local/ 运行证据", 13, INK, True), ("+ sediment_staging", 11, INK, False),
                ("（Tier1 候选）", 10, MUT, False)])
    D = box(s, 10.45, by, 2.35, bh, L_AMBER, line=AMBER, radius=0.1)
    settext(D, [("Curator + CI 门", 13, AMBER, True), ("去重 · 冲突 · eval-gate", 10, INK, False),
                ("脱敏 · Pareto", 10, INK, False)])
    midy = by + bh / 2
    for x1, x2, lab, cx in [(3.30, 3.85, "① 消费", 3.575), (6.80, 7.35, "② 蒸馏", 7.075),
                            (9.90, 10.45, "③ 沉淀PR", 10.175)]:
        arrow(s, x1, midy, x2, midy, color=INK, weight=2.5)
        badge(s, cx, by - 0.32, lab, color=INDIGO, w=1.25)
    # loop back
    la = shape(s, MSO_SHAPE.LEFT_ARROW, 0.6, 4.75, 12.2, 0.72, L_BLUE, line=INDIGO)
    settext(la, [("④ 发布：eval-gate 通过才回灌 pipeline（闭环·pinned 版本）", 13, INDIGO, True)])
    takeaway(s, 5.9, "每一环都有质量门，脏数据不滚雪球；hub 以 pinned 版本反向喂回，可复现、可回滚。")
    footer(s)

    # S3 two engines -------------------------------------------------------
    s = prs.slides.add_slide(blank)
    header(s, "脊柱：两类资产 · 两套合并引擎", "用同一套 git 行级合并治理两者 = 团队记忆仓失败根因；分开走", "03")
    # left: Skills / engine B
    box(s, 0.55, 1.45, 6.0, 4.05, L_INDIGO, line=INDIGO, radius=0.05)
    textbox(s, 0.7, 1.55, 5.7, 0.4, [("Skills（过程性）— 引擎 B：验证门竞争式编辑", 13.5, INDIGO, True)])
    e1 = box(s, 0.85, 2.10, 5.4, 0.6, WHITE, line=LINE)
    settext(e1, [("成员提议：有界编辑 add / del / replace（文本学习率裁剪）", 10.5, INK, False)])
    e2 = box(s, 1.55, 2.95, 4.0, 0.6, L_AMBER, line=AMBER)
    settext(e2, [("留出 eval 套件：严格变好？", 11, AMBER, True)])
    arrow(s, 3.55, 2.70, 3.55, 2.95, INK, 2)
    e3 = box(s, 0.85, 3.80, 2.55, 0.62, L_GREEN, line=GREEN)
    settext(e3, [("是 → 更新", 10, GREEN, True), ("best_skill.md + scorecard", 9.5, INK, False)])
    e4 = box(s, 3.70, 3.80, 2.55, 0.62, RGBColor(0xFD, 0xE7, 0xE9), line=RED)
    settext(e4, [("否 → 进 bad_edits", 10, RED, True), ("缓冲（不再重试）", 9.5, INK, False)])
    arrow(s, 2.7, 3.55, 2.1, 3.80, GREEN, 1.75)
    arrow(s, 4.4, 3.55, 5.0, 3.80, RED, 1.75)
    e5 = box(s, 0.85, 4.65, 5.4, 0.66, RGBColor(0xE6, 0xE2, 0xFF), line=INDIGO)
    settext(e5, [("GEPA Pareto：保留互补候选，防多人编辑塌缩到局部最优", 10.5, INDIGO, True)])
    # right: Knowledge / engine A
    box(s, 6.78, 1.45, 6.0, 4.05, L_ORANGE, line=ORANGE, radius=0.05)
    textbox(s, 6.93, 1.55, 5.7, 0.4, [("Knowledge（事实/记忆）— 引擎 A：集合并 + 去重 + 冲突消解", 13, ORANGE, True)])
    k1 = box(s, 7.05, 2.10, 5.5, 0.55, WHITE, line=LINE)
    settext(k1, [("稳定 ID · 集合并（绝不 git 行级合并）", 11, INK, True)])
    k2 = box(s, 7.05, 2.85, 5.5, 0.6, WHITE, line=LINE)
    settext(k2, [("近似重复？→ 合并出处，confirmations + 1", 10.5, INK, False)])
    k3 = box(s, 7.05, 3.60, 5.5, 0.7, WHITE, line=LINE)
    settext(k3, [("矛盾？→ 证据/新近度加权（Zep 双时态）", 10.5, INK, False),
                 ("旧记录标 superseded，不删除，可审计", 9.5, MUT, False)])
    k4 = box(s, 7.05, 4.45, 5.5, 0.55, WHITE, line=LINE)
    settext(k4, [("否则 → 新增（带出处 + 证据，无出处即拒）", 10.5, INK, False)])
    for yy in (2.65, 3.45, 4.30):
        arrow(s, 9.8, yy, 9.8, yy + 0.15, ORANGE, 1.75)
    takeaway(s, 5.75, "知识靠『集合并 + 去重 + 冲突消解』；技能靠『验证门竞争式编辑 + Pareto』。两类资产，两台引擎。")
    footer(s)

    # S4 funnel ------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    header(s, "沉淀漏斗：三层 · 三道门 · L0–L3", "一条原始轨迹如何层层过门、晋升为团队资产", "04")
    rows = [
        (2.4, 8.5, L_GRAY, LINE, [("Tier0 运行轨迹", 12.5, INK, True),
            ("plans · reviews · bench · design（本地，可能含密钥）", 10, MUT, False)]),
        (3.1, 7.6, L_BLUE, BLUE, [("Tier1 候选 = L1 — 类型化 + 出处 + 证据（staging）", 11.5, BLUE, True)]),
        (3.7, 6.8, L_AMBER, AMBER, [("门1 · Schema / Lint / 脱敏", 11.5, AMBER, True)]),
        (3.7, 6.0, L_AMBER, AMBER, [("门2 · 证据门：引用 · delta · ≥N 确认", 11.5, AMBER, True)]),
        (3.7, 5.2, L_AMBER, AMBER, [("门3 · 策展 + eval：Curator + 双评审 + eval-gate", 11.5, AMBER, True)]),
        (3.7, 4.4, L_GREEN, GREEN, [("Tier2 = L2 stable → knowledge/ 或 skills/domain/", 11.5, GREEN, True)]),
        (4.5, 3.4, RGBColor(0xC7, 0xEC, 0xD6), GREEN, [("L3 core → skills/core/（组织金标准）", 11.5, GREEN, True)]),
    ]
    ys = [1.35, 2.15, 2.95, 3.75, 4.55, 5.35, 6.15]
    cx = EW / 2
    for (yy, (rh, w, fill, ln, lines)) in zip(ys, [(0.6, *r[1:]) for r in rows]):
        b = box(s, cx - w / 2, yy, w, 0.6, fill, line=ln)
        settext(b, lines)
    labels = ["收口点蒸馏 hmopt sediment", "", "", "", "", "跨子团队复用成功"]
    for i in range(6):
        y1 = ys[i] + 0.6
        y2 = ys[i + 1]
        arrow(s, cx, y1, cx, y2, INK, 2)
        if labels[i]:
            textbox(s, cx + 0.2, (y1 + y2) / 2 - 0.16, 3.6, 0.32, [(labels[i], 9.5, MUT, False)])
    # reject side-notes
    textbox(s, 10.7, 2.95, 2.4, 0.9,
            [("拒 → 打回 / 留本地", 10, RED, True), ("无证据 → 留 L1", 10, RED, False),
             ("破例 → 降级 + 签字", 10, RED, False), ("（无豁免口子）", 9, MUT, False)])
    footer(s)

    # S5 skills layout -----------------------------------------------------
    s = prs.slides.add_slide(blank)
    header(s, "skills/ 内部布局：消灭组合爆炸", "只有 3 个维度是真正的 skill 文件夹；file / function → knowledge", "05")
    dims = ["流程 / 跨切面", "优化招式 mechanism", "子系统 subsystem", "目录 dir", "文件 file", "函数 function / symbol"]
    dy = 1.45
    dim_boxes = []
    for d in dims:
        b = box(s, 0.6, dy, 3.0, 0.62, L_GRAY, line=LINE)
        settext(b, [(d, 11, INK, True)], align=PP_ALIGN.LEFT)
        dim_boxes.append((b, dy + 0.31))
        dy += 0.78
    # skill targets (blue)
    t_core = box(s, 8.3, 1.5, 4.4, 0.7, L_BLUE, line=BLUE)
    settext(t_core, [("skills/core/   流程·跨切面", 11.5, BLUE, True)])
    t_tech = box(s, 8.3, 2.45, 4.4, 0.7, L_BLUE, line=BLUE)
    settext(t_tech, [("skills/technique/   优化招式", 11.5, BLUE, True)])
    t_dom = box(s, 8.3, 3.40, 4.4, 0.7, L_BLUE, line=BLUE)
    settext(t_dom, [("skills/domain/ 按子系统/  ← 唯一拓扑层", 11, BLUE, True)])
    # knowledge targets (orange)
    k_f = box(s, 8.3, 4.65, 4.4, 0.7, L_ORANGE, line=ORANGE)
    settext(k_f, [("knowledge/targets/facts/", 11.5, ORANGE, True)])
    k_l = box(s, 8.3, 5.60, 4.4, 0.7, L_ORANGE, line=ORANGE)
    settext(k_l, [("knowledge/ idea_ledger + symbol_selectors", 10.5, ORANGE, True)])
    targets = {0: (t_core, 1.85, BLUE, False), 1: (t_tech, 2.80, BLUE, False), 2: (t_dom, 3.75, BLUE, False),
               3: (t_dom, 3.75, MUT, True), 4: (k_f, 5.0, ORANGE, False), 5: (k_l, 5.95, ORANGE, False)}
    for i, (db, dcy) in enumerate(dim_boxes):
        tb, tcy, col, dash = targets[i]
        arrow(s, 3.6, dcy, 8.3, tcy, col, 1.75, dash=dash)
    textbox(s, 4.0, 3.95, 4.2, 0.5, [("目录: applies_to.path_globs", 9.5, MUT, True), ("（不单独建目录）", 9, MUT, False)])
    takeaway(s, 6.55, "组合而非枚举：resolver 按 selector 解析 domain，requires 拉入 core+technique，knowledge 按符号检索挂载。")
    footer(s)

    # S6 roadmap -----------------------------------------------------------
    s = prs.slides.add_slide(blank)
    header(s, "分阶段路线图", "Phase 3（建 eval 套件）是关键长杆，红色标注", "06")
    phases = [
        ("Phase 0\n抽取\n1–2w", INDIGO, "双仓 pin + 路径兼容"),
        ("Phase 1\n蒸馏\n2–3w", BLUE, "hmopt sediment"),
        ("Phase 2\n策展+合并\n3–6w", TEAL, "引擎A + CI + policies"),
        ("Phase 3\neval 门 ★\n6–10w", RED, "core suite + 引擎B"),
        ("Phase 4\n自动优化\n10w+", GREEN, "定时作业 + 发布节奏"),
    ]
    cw, cx0, cyy = 2.62, 0.45, 2.55
    for i, (title, col, deliv) in enumerate(phases):
        x = cx0 + i * (cw - 0.18)
        ch = shape(s, MSO_SHAPE.CHEVRON, x, cyy, cw, 1.7, col)
        lines = [(t, 13 if j == 0 else 11, WHITE, j == 0) for j, t in enumerate(title.split("\n"))]
        settext(ch, lines)
        d = box(s, x + 0.15, cyy + 1.95, cw - 0.2, 0.85, L_GRAY, line=LINE)
        settext(d, [(deliv, 10.5, INK, False)])
    takeaway(s, 5.95, "先 symlink 兜底跑通双仓（Phase 0）；eval 套件最难也最值；优化作业半自动 → 全自动。")
    footer(s)

    out = Path(__file__).resolve().parent / "Team_Skill_Hub_Design_Slides.pptx"
    prs.save(str(out))
    print("saved:", out)
    return out


if __name__ == "__main__":
    build()
