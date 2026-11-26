"""Figures 2 and 3 of the paper. Run from anywhere: python scripts/figures/fig23.py
(first run scripts/steer/analyse_exp06.py). Writes figures/figure2.{pdf,png} and figures/figure3.{pdf,png}."""
import json, numpy as np, matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8.5, "legend.fontsize": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
ROOT = Path(__file__).resolve().parents[2]  # repository root
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
# both files are written by scripts/steer/analyse_exp06.py
D = json.load(open(ROOT / "results/exp06/analysis/deltas.json", encoding="utf-8"))
A = json.load(open(ROOT / "results/exp06/analysis/analysis.json", encoding="utf-8"))
L = ["breathy", "belt", "vocal_fry", "forte", "piano"]
N = {"breathy": "Breathy", "belt": "Belt", "vocal_fry": "Vocal fry", "forte": "Forte", "piano": "Piano"}
conds = [("eqmatch", "Matched EQ", "s", "#8c8c8c", "#8c8c8c"),
         ("oracleeq", "Oracle EQ (given the real spectrum)", "D", "#a6a6a6", "white"),
         ("a1", "Adapter, α = 1", "o", "#1f5fa8", "#1f5fa8")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, m, title in ((axes[0], "d_static", "Static part, ΔD$_{\\mathrm{static}}$"),
                     (axes[1], "d_dyn", "Equaliser-invariant part, ΔD$_{\\mathrm{dyn}}$")):
    ax.axvline(0, color="#4d4d4d", lw=0.8, zorder=1)
    for i, k in enumerate(L):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f2f2f2", zorder=0, lw=0)
        for j, (c, lab, mk, col, face) in enumerate(conds):
            est, lo, hi = D[f"{k}|{c}|{m}"]
            y = i + (j - 1) * 0.24
            ax.plot([lo, hi], [y, y], color=col, lw=1.3, zorder=2, solid_capstyle="butt")
            ax.plot(est, y, marker=mk, ms=5.2 if mk != "D" else 4.6, mfc=face, mec=col, mew=1.2, ls="none", zorder=3,
                    label=lab if i == 0 else None)
    ax.set_title(title, loc="left", pad=4)
    ax.set_xlabel("Change relative to the unmodified output (dB)")
    ax.grid(axis="x", color="#e0e0e0", lw=0.6, zorder=0)
    ax.set_ylim(len(L) - 0.5, -0.5)
axes[0].set_yticks(range(len(L)))
axes[0].set_yticklabels([N[k] for k in L])
axes[0].tick_params(axis="y", length=0)
axes[1].tick_params(axis="y", length=0)
axes[0].set_xlim(-5.2, 1.2)
axes[1].set_xlim(-2.2, 0.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5)
fig.savefig(OUT / "figure3.pdf", bbox_inches="tight")
fig.savefig(OUT / "figure3.png", dpi=220, bbox_inches="tight")

# Figure 2: closest label among the five, by D and by D_dyn
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
for ax, metric, title in ((axes[0], "d_raw", "Closest label by D"), (axes[1], "d_dyn", "Closest label by D$_{\\mathrm{dyn}}$")):
    w = A["confusion"][metric]["closest_label_counts"]
    M = np.array([[w[k][j] for j in L] for k in L], float)
    n = np.array([w[k]["n"] for k in L], float)
    P = M / n[:, None]
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for a in range(5):
        for b in range(5):
            ax.text(b, a, f"{int(M[a, b])}", ha="center", va="center", fontsize=8.5,
                    color="white" if P[a, b] > 0.55 else "#1a1a1a", fontweight="bold" if a == b else "normal")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([N[k] for k in L], rotation=30, ha="right")
    if metric == "d_raw":
        ax.set_yticklabels([f"{N[k]} (n={int(n[i])})" for i, k in enumerate(L)])
    ax.set_xlabel("Adapter label (α = 1)")
    ax.set_title(title, loc="left", pad=4)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
axes[0].set_ylabel("Real technique recording")
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("Share of pairs in the row")
cb.outline.set_visible(False)
fig.savefig(OUT / "figure2.pdf", bbox_inches="tight")
fig.savefig(OUT / "figure2.png", dpi=220, bbox_inches="tight")
print("ok")
