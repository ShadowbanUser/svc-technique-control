"""Figure 4 of the paper. Run from anywhere: python scripts/figures/fig4.py (first run scripts/steer/analyse_exp06.py).

(a), (b): listener-level shares with 95% bootstrap intervals over listeners (4000 resamples, seed 20261125). They need
the per-listener responses, which are not public (listeners were told results would be published only in anonymous,
aggregated form). If that file is absent, the values are read from figures/figure4_values.json instead.
(c): adapter minus its matched equaliser in D_dyn, read from results/exp06/analysis/analysis.json, i.e. exactly the
numbers of Table 7, column 2 (cluster bootstrap over the five singers, 2000 resamples, seed 20261119; Section 5.6).
Writes figures/figure4.{pdf,png} and figures/figure4_values.json.
"""
import csv, json, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8.5, "legend.fontsize": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
TECH = [("breathy", "Breathy"), ("vocal_fry", "Vocal fry"), ("belt", "Belt"), ("forte", "Forte"), ("piano", "Piano")]
INT = {"breathy": 3, "belt": 4, "vocal_fry": 5, "forte": 6, "piano": 7}
BLUE, GREY = "#1f5fa8", "#8c8c8c"
ROOT = Path(__file__).resolve().parents[2]  # repository root
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
PRIVATE = ROOT / "results/listening_main/analysis_final_2026-11-25/responses_long.csv"  # author only
rng = np.random.default_rng(20261125)

def boot(per_listener, stat, B=4000):
    """per_listener: {listener: list of items}; stat(list_of_lists) -> value. Bootstrap over listeners."""
    keys = list(per_listener); vals = [per_listener[k] for k in keys]
    est = stat(vals); bs = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys)); bs.append(stat([vals[i] for i in pick]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return est, lo, hi

def share(lists):
    flat = [x for l in lists for x in l]; return np.mean(flat) if flat else np.nan

pref, ident, real = {}, {}, {}
if PRIVATE.exists():
    rows = list(csv.DictReader(open(PRIVATE, encoding="utf-8")))
    for t, _ in TECH:
        d = collections.defaultdict(list); i = collections.defaultdict(list); rr = collections.defaultdict(list)
        for r in rows:
            if r["technique"] != t:
                continue
            if r["type"] == "E" and r["condition"] == "adapter" and r["more_technique"] in ("neural", "eq"):
                d[r["listener"]].append(r["more_technique"] == "neural")
            if r["type"] == "I" and r["change"]:
                (i if r["condition"] == "adapter" else rr if r["condition"] == "real" else {})[r["listener"]].append(int(r["change"]) == INT[t]) if r["condition"] in ("adapter", "real") else None
        pref[t] = boot(d, share); ident[t] = boot(i, share); real[t] = boot(rr, share)
else:
    saved = json.load(open(OUT / "figure4_values.json", encoding="utf-8"))
    for t, _ in TECH:
        pref[t], ident[t], real[t] = saved[t]["pref"], saved[t]["ident"], saved[t]["real"]

# objective: adapter minus its matched EQ in D_dyn = Table 7, column 2 (written by scripts/steer/analyse_exp06.py)
A = json.load(open(ROOT / "results/exp06/analysis/analysis.json", encoding="utf-8"))
obj = {t: tuple(A["groundtruth"][t]["adapter_minus_eqmatch"]["d_dyn"]) for t, _ in TECH}

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True, gridspec_kw={"wspace": 0.28})
ys = np.arange(len(TECH))[::-1]
def band(ax):
    for k, y in enumerate(ys):
        if k % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f2f2f2", zorder=0, lw=0)
    ax.set_ylim(-0.6, len(TECH) - 0.4)
def subtitle(ax, text):  # material and strength of the panel
    ax.text(0, 1.012, text, transform=ax.transAxes, fontsize=7.5, color="#4d4d4d", ha="left", va="bottom")
def dot(ax, y, e, col, face, mk="o"):
    ax.plot([e[1], e[2]], [y, y], color=col, lw=1.3, zorder=2, solid_capstyle="butt")
    ax.plot(e[0], y, mk, ms=5.5, mfc=face, mec=col, mew=1.2, zorder=3)

ax = axes[0]; band(ax)
ax.axvline(0.5, color="#4d4d4d", lw=0.8, ls="--", zorder=1)
for y, (t, _) in zip(ys, TECH):
    dot(ax, y, pref[t], BLUE, BLUE)
ax.set_xlim(0.3, 1.04); ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.set_xlabel("Share choosing the control\n(of decided responses)")
ax.set_title("(a) Control vs. matched EQ", loc="left", pad=13); subtitle(ax, "Sung phrases, α = 1 (fry 0.6)")
ax.set_yticks(ys); ax.set_yticklabels([n for _, n in TECH])

ax = axes[1]; band(ax)
ax.axvline(1 / 9, color="#4d4d4d", lw=0.8, ls="--", zorder=1)
off = 0.14
for y, (t, _) in zip(ys, TECH):
    dot(ax, y + off, ident[t], BLUE, BLUE)
    dot(ax, y - off, real[t], GREY, "white", "D")
ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", ".25", ".5", ".75", "1"]); ax.set_xlabel("Share naming the\nintended technique")
ax.set_title("(b) Identification", loc="left", pad=13); subtitle(ax, "Control: sung phrases, α = 0.5 to 1")
ax.plot([], [], "o", mfc=BLUE, mec=BLUE, label="Control"); ax.plot([], [], "D", mfc="white", mec=GREY, label="Real recording")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.31), ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.0)

ax = axes[2]; band(ax)
ax.axvline(0, color="#4d4d4d", lw=0.8, zorder=1)
for y, (t, _) in zip(ys, TECH):
    dot(ax, y, obj[t], BLUE, BLUE)
ax.set_xlabel("$D_{\\mathrm{dyn}}$: control minus\nmatched EQ (dB)")
ax.set_title("(c) Objective, $D_{\\mathrm{dyn}}$", loc="left", pad=13); subtitle(ax, "VocalSet exercises, α = 1")
for a in axes:
    a.grid(axis="x", color="#e0e0e0", lw=0.6, zorder=0); a.tick_params(axis="y", length=0)
fig.savefig(OUT / "figure4.pdf", bbox_inches="tight"); fig.savefig(OUT / "figure4.png", dpi=300, bbox_inches="tight")
out = {t: {"pref": pref[t], "ident": ident[t], "real": real[t], "obj": obj[t]} for t, _ in TECH}
json.dump({t: {k: [round(float(x), 3) for x in v] for k, v in d.items()} for t, d in out.items()}, open(OUT / "figure4_values.json", "w", newline="\n"), indent=1)
print(json.dumps(json.load(open(OUT / "figure4_values.json")), indent=0))
