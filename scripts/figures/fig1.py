import cairosvg
W, H = 1000, 1010
ACC = "#1f5fa8"; ACC_FILL = "#e6eef8"; INK = "#1a1a1a"; QUIET = "#555555"; EDGE = "#6e6e6e"; FROZEN = "#f0f0f0"
out = []
def add(s): out.append(s)
def text(x, y, s, size=18.4, weight="normal", anchor="middle", fill=INK, style="normal"):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    add(f'<text x="{x}" y="{y}" font-family="DejaVu Sans" font-size="{size}" font-weight="{weight}" font-style="{style}" text-anchor="{anchor}" fill="{fill}">{s}</text>')
def box(x, y, w, h, lines, fill="white", stroke=EDGE, sw=1.4, size=17.3, bold_first=False, color=INK, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
    n = len(lines); lh = size * 1.3
    y0 = y + h / 2 - (n - 1) * lh / 2 + size * 0.35
    for i, ln in enumerate(lines):
        text(x + w / 2, y0 + i * lh, ln, size=size, weight="bold" if (bold_first and i == 0) else "normal", fill=color)
def arrow(x1, y1, x2, y2, color=EDGE, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    mid = "arrA" if color == ACC else "arr"
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#{mid})"{d}/>')
def path(d, color=EDGE, sw=1.6, dash=None):
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    mid = "arrA" if color == ACC else "arr"
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#{mid})"{dd}/>')

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#6e6e6e"/></marker>'
    f'<marker id="arrA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{ACC}"/></marker></defs>')
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>')
# panel frames and titles
panels = [(15, "(a) Parallel technique pairs (training)"), (515, "(b) Technique-gated low-rank adapter")]
for x, t in panels:
    add(f'<rect x="{x}" y="12" width="470" height="616" rx="10" fill="none" stroke="#c8c8c8" stroke-width="1.2"/>')
    text(x + 16, 44, t, size=18, weight="bold", anchor="start")

# ---------------- panel (a)
box(40, 72, 200, 62, ["Plain take", "singer s, exercise e"], size=16.7)
box(265, 72, 200, 62, ["Technique take k", "singer s, exercise e"], size=16.7)
arrow(140, 134, 140, 168); arrow(365, 134, 365, 168)
box(40, 170, 200, 62, ["Content (Whisper)", "and F0 (RMVPE)"], size=16.7)
box(265, 170, 200, 62, ["Pitch, voicing,", "energy of both takes"], size=16.7)
arrow(365, 232, 365, 270)
box(160, 272, 305, 70, ["Dynamic time warping", "key offset removed"], size=16.7)
arrow(100, 232, 100, 382)
path("M 300 342 L 300 362 L 180 362 L 180 382")
box(40, 384, 250, 70, ["Plain content and F0", "on the technique time line"], size=16.7)
arrow(385, 342, 385, 382)
box(305, 384, 160, 70, ["Mel of the", "technique take"], size=16.7)
arrow(165, 454, 165, 488); arrow(385, 454, 385, 488)
box(40, 490, 425, 114, ["Training example", "input: warped plain content and F0", "prompt: another plain take of singer s", "target: technique mel, one-hot label k", "949 training pairs from 15 singers"], size=15.7, bold_first=True)

# ---------------- panel (b)
box(540, 72, 205, 62, ["Condition", "content + F0 of source"], size=16.2)
box(760, 72, 205, 62, ["Reference", "prompt + style vector"], size=16.2)
arrow(642, 134, 642, 168); arrow(862, 134, 862, 168)
add(f'<rect x="540" y="170" width="425" height="330" rx="10" fill="{FROZEN}" stroke="{EDGE}" stroke-width="1.4"/>')
text(752, 196, "Frozen Seed-VC decoder (DiT, 17 layers, 186 M)", size=15, weight="bold")
text(752, 218, "96 wrapped layers; 9.8 M trainable parameters", size=14.6, fill=QUIET)
add(f'<rect x="560" y="232" width="385" height="254" rx="8" fill="white" stroke="{EDGE}" stroke-width="1.2"/>')
text(752, 255, "One wrapped linear layer", size=15.1, fill=QUIET)
text(752, 284, "y = W x + B (g \u2299 A x),   g = \u03a3\u2096 \u03b1\u2096 e\u2096", size=17.8, style="italic")
text(752, 306, "A, B shared by all techniques; one gate e\u2096 each", size=14.6, fill=QUIET)
box(590, 322, 105, 46, ["x"], size=18.4)
box(805, 322, 120, 46, ["W x  (frozen)"], size=16.2, fill=FROZEN)
arrow(695, 345, 803, 345)
box(590, 412, 72, 44, ["A"], size=18.4, fill=ACC_FILL, stroke=ACC, sw=2)
box(690, 412, 92, 44, ["\u2299 g"], size=18.4, fill=ACC_FILL, stroke=ACC, sw=2)
box(810, 412, 72, 44, ["B"], size=18.4, fill=ACC_FILL, stroke=ACC, sw=2)
arrow(626, 368, 626, 410, color=ACC)
arrow(662, 434, 688, 434, color=ACC); arrow(782, 434, 808, 434, color=ACC)
path("M 882 434 L 905 434 L 905 370", color=ACC)
box(540, 516, 425, 90, ["Technique strengths \u03b1, one per technique", "breathy, belt, vocal fry, forte, piano", "continuous and combinable", "\u03b1 = 0: branch skipped, output bit-identical"], size=15.1, fill=ACC_FILL, stroke=ACC, sw=2)
path("M 736 516 L 736 458", color=ACC)


# ---------------- panel (c), second row
Y = 640
add(f'<rect x="15" y="{Y}" width="970" height="358" rx="10" fill="none" stroke="#c8c8c8" stroke-width="1.2"/>')
text(33, Y + 32, "(c) Evaluation against real recordings", size=18, weight="bold", anchor="start")
box(35, Y + 58, 200, 64, ["Plain take", "held-out singer s"], size=17)
box(35, Y + 200, 200, 64, ["Real technique take", "same singer, exercise"], size=17)
box(268, Y + 55, 215, 70, ["Converter + adapter", "reference: s, plain"], size=17, fill=ACC_FILL, stroke=ACC, sw=2)
box(268, Y + 197, 215, 70, ["DTW path, plain ↔", "technique (fixed)"], size=17, dash="6 4")
box(516, Y + 58, 150, 64, ["Output", "plain time line"], size=17)
arrow(235, Y + 90, 266, Y + 90); arrow(483, Y + 90, 514, Y + 90)
arrow(235, Y + 232, 266, Y + 232)
path(f"M 135 {Y + 122} L 135 {Y + 160} L 375 {Y + 160} L 375 {Y + 195}", dash="5 4")
box(700, Y + 50, 270, 158, ["Envelope distance D", "30 cepstra of the log-mel", "D² = D_static² + D_dyn²", "fixed per-band gain", "changes only D_static", "vs. matched, oracle EQ"], size=16, bold_first=True)
arrow(666, Y + 90, 698, Y + 90)
path(f"M 483 {Y + 232} L 600 {Y + 232} L 600 {Y + 175} L 698 {Y + 175}")
box(700, Y + 236, 270, 96, ["Technique classifier", "trained on real recordings", "dynamic vs. full features"], size=16, bold_first=True)
arrow(835, Y + 208, 835, Y + 234)
add('</svg>')
svg = "\n".join(out)
open("figure1.svg", "w").write(svg)
cairosvg.svg2png(bytestring=svg.encode(), write_to="figure1.png", output_width=2000)
cairosvg.svg2pdf(bytestring=svg.encode(), write_to="figure1.pdf")
print("ok")
