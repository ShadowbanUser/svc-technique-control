"""Analysis of experiment 06 (both parts) for the paper. NumPy only; reads the result files.

Inputs   results/exp06/groundtruth/{metrics,conditions}.json
         results/exp06/classifier/{summary.json,predictions.jsonl}
Outputs  results/exp06/analysis/{analysis.json,deltas.json,report.md}
         analysis.json holds every number of Tables 6 to 8 and Figures 2 and 3 of the paper; deltas.json holds
         the per-condition changes relative to the unmodified output drawn in Figure 3; Table 7, column 2
         (adapter minus its matched equaliser, D_dyn) is groundtruth/<technique>/adapter_minus_eqmatch/d_dyn.
         With --exact, every interval is computed from all n^n ordered resamples of the n singers instead of
         2000 random resamples (written to results/exp06/analysis_exact/, for comparison only).

Pre-specified checks (paper Section 3, written before the results):
  O1  breathy, vocal_fry: D_dyn(adapter, correct label) < D_dyn(base), and the correct label
      lowers D_dyn more than the mean of the four wrong labels; the dyn-feature classifier
      gives the intended technique a higher probability than for the base output.
  O2  belt, forte, piano: most of the drop in D^2 from base to adapter lies in the static part.
Uncertainty: cluster bootstrap over the five held-out singers (singers resampled with
replacement, all items of a drawn singer kept), 95% percentile intervals over 2000 resamples
drawn with numpy.random.default_rng(20261119); these are the intervals reported in the paper.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ("breathy", "belt", "vocal_fry", "forte", "piano")
NAMES = {"breathy": "Breathy", "belt": "Belt", "vocal_fry": "Vocal fry", "forte": "Forte", "piano": "Piano"}
B = 2000
SEED = 20261119
EXACT = False  # set by --exact


def _exact_draws(singers, stat_of_pick):
    if len(singers) > 7:
        raise ValueError("exact enumeration is only practical for a handful of clusters")
    return [stat_of_pick(p) for p in itertools.product(range(len(singers)), repeat=len(singers))]


def boot(values_by_singer, stat=np.mean, n=B, seed=SEED):
    """values_by_singer: {singer: array}. Returns (estimate, lo, hi)."""
    singers = sorted(values_by_singer)
    allv = np.concatenate([np.asarray(values_by_singer[s], float) for s in singers])
    est = float(stat(allv))
    if EXACT:
        draws = _exact_draws(singers, lambda pick: stat(np.concatenate(
            [np.asarray(values_by_singer[singers[i]], float) for i in pick])))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return est, float(lo), float(hi)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rng.choice(len(singers), size=len(singers), replace=True)
        draws.append(stat(np.concatenate([np.asarray(values_by_singer[singers[i]], float) for i in pick])))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return est, float(lo), float(hi)


def ratio_boot(num_by_singer, den_by_singer, n=B, seed=SEED):
    singers = sorted(num_by_singer)
    num = {s: np.asarray(num_by_singer[s], float) for s in singers}
    den = {s: np.asarray(den_by_singer[s], float) for s in singers}
    est = float(np.concatenate(list(num.values())).mean() / np.concatenate(list(den.values())).mean())

    def one(pick):
        a = np.concatenate([num[singers[i]] for i in pick]).mean()
        b = np.concatenate([den[singers[i]] for i in pick]).mean()
        return a / b if b != 0 else np.nan
    if EXACT:
        lo, hi = np.nanpercentile(_exact_draws(singers, one), [2.5, 97.5])
        return est, float(lo), float(hi)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rng.choice(len(singers), size=len(singers), replace=True)
        a = np.concatenate([num[singers[i]] for i in pick]).mean()
        b = np.concatenate([den[singers[i]] for i in pick]).mean()
        draws.append(a / b if b != 0 else np.nan)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return est, float(lo), float(hi)


def fmt(e, nd=2):
    if e is None:
        return "–"
    if isinstance(e, tuple):
        return f"{e[0]:+.{nd}f} [{e[1]:+.{nd}f}, {e[2]:+.{nd}f}]"
    return f"{e:.{nd}f}"


def load_items(gt_dir):
    rows = json.loads((gt_dir / "metrics.json").read_text(encoding="utf-8"))
    items = {}
    for r in rows:
        if r.get("d_raw") is None:
            continue
        r["d_dyn"] = r["d_cmn"]
        r["d_static"] = float(np.sqrt(max(r["d_raw"] ** 2 - r["d_cmn"] ** 2, 0.0)))
        items.setdefault((r["singer"], r["sid"], r["truth"]), {})[r["condition"]] = r
    return items


def per_singer(items, truth, fn):
    """fn(conds) -> value or None; grouped by singer for items of this truth technique."""
    out = {}
    for (singer, sid, t), conds in items.items():
        if t != truth:
            continue
        v = fn(conds)
        if v is not None and np.isfinite(v):
            out.setdefault(singer, []).append(v)
    return out


def groundtruth(items):
    res = {}
    for k in LABELS:
        r = {}
        ad = f"ad_{k}_a1"
        wrong = [f"ad_{j}_a1" for j in LABELS if j != k]
        get = lambda c, m: (lambda conds: conds[c][m] if c in conds else None)  # noqa: E731

        def mean_of(cond, metric):
            g = per_singer(items, k, get(cond, metric))
            return boot(g) if g else None

        conds = {"real_straight": "real_straight", "base": "base", "adapter_a0.5": f"ad_{k}_a0.5", "adapter_a1": ad,
                 "eqmatch": f"eqmatch_{k}", "oracleeq": f"oracleeq_{k}", "guidance": f"cgm_{k}",
                 "carried": f"carried_{k}", "real_straight_oracleeq": f"real_straight_oracleeq_{k}"}
        r["means"] = {name: {m: mean_of(c, m) for m in ("d_raw", "d_static", "d_dyn")} for name, c in conds.items()}
        r["means"]["static_bound"] = {"d_raw": mean_of("base", "d_dyn")}

        def diff(c1, c0, metric):
            return lambda conds: (conds[c1][metric] - conds[c0][metric]) if c1 in conds and c0 in conds else None
        r["adapter_minus_base"] = {m: boot(per_singer(items, k, diff(ad, "base", m))) for m in ("d_raw", "d_static", "d_dyn")}
        r["adapter_minus_eqmatch_raw"] = boot(per_singer(items, k, diff(ad, f"eqmatch_{k}", "d_raw")))
        # Table 7, column 2 uses the D_dyn entry
        r["adapter_minus_eqmatch"] = {m: boot(per_singer(items, k, diff(ad, f"eqmatch_{k}", m)))
                                      for m in ("d_raw", "d_static", "d_dyn")}
        # Figure 3: change of each condition relative to the unmodified output (also written to deltas.json)
        r["change_from_base"] = {arm: {m: boot(per_singer(items, k, diff(c, "base", m))) for m in ("d_raw", "d_static", "d_dyn")}
                                 for arm, c in (("eqmatch", f"eqmatch_{k}"), ("oracleeq", f"oracleeq_{k}"), ("a1", ad),
                                                ("a0.5", f"ad_{k}_a0.5"), ("carried", f"carried_{k}"))}
        r["adapter_minus_oracleeq_raw"] = boot(per_singer(items, k, diff(ad, f"oracleeq_{k}", "d_raw")))
        r["adapter_raw_minus_static_bound"] = boot(per_singer(
            items, k, lambda c: c[ad]["d_raw"] - c["base"]["d_dyn"] if ad in c else None))
        guid = per_singer(items, k, diff(f"cgm_{k}", "base", "d_dyn"))
        r["guidance_minus_base_dyn"] = boot(guid) if guid else None

        def spec(metric):
            def f(c):
                if ad not in c or not all(w in c for w in wrong):
                    return None
                return (c[ad][metric] - c["base"][metric]) - np.mean([c[w][metric] - c["base"][metric] for w in wrong])
            return f
        r["correct_minus_wrong"] = {m: boot(per_singer(items, k, spec(m))) for m in ("d_raw", "d_dyn")}
        num = per_singer(items, k, lambda c: (c["base"]["d_static"] ** 2 - c[ad]["d_static"] ** 2) if ad in c else None)
        den = per_singer(items, k, lambda c: (c["base"]["d_raw"] ** 2 - c[ad]["d_raw"] ** 2) if ad in c else None)
        r["static_share_of_drop"] = ratio_boot(num, den)
        r["mean_drop_raw_sq"] = float(np.concatenate([np.asarray(v) for v in den.values()]).mean())
        gap_n = per_singer(items, k, lambda c: (c["base"]["d_raw"] - c[ad]["d_raw"]) if ad in c and f"carried_{k}" in c else None)
        gap_d = per_singer(items, k, lambda c: (c["base"]["d_raw"] - c[f"carried_{k}"]["d_raw"]) if ad in c and f"carried_{k}" in c else None)
        r["gap_closed_toward_carried"] = ratio_boot(gap_n, gap_d) if gap_n else None
        for m in ("cpp_db_minus_truth", "hf_ratio_db_minus_truth"):
            r[m] = {name: boot(per_singer(items, k, get(c, m))) for name, c in
                    (("base", "base"), ("adapter_a1", ad), ("carried", f"carried_{k}"))
                    if per_singer(items, k, get(c, m))}
        r["var_ratio"] = {name: boot(per_singer(items, k, get(c, "var_ratio"))) for name, c in
                          (("base", "base"), ("adapter_a1", ad), ("carried", f"carried_{k}"))}
        r["n_items"] = sum(1 for (s, sid, t) in items if t == k)
        low = per_singer(items, k, lambda c: float(c[ad]["d_raw"] < c["base"]["d_raw"]) if ad in c else None)
        r["share_of_pairs_adapter_lowers_d"] = float(np.concatenate([np.asarray(v) for v in low.values()]).mean())
        r["per_singer_adapter_minus_base_dyn"] = {s: float(np.mean(v)) for s, v in
                                                  per_singer(items, k, diff(ad, "base", "d_dyn")).items()}
        res[k] = r
    conf = {}
    for metric in ("d_raw", "d_dyn"):
        mat, wins = {}, {}
        for k in LABELS:
            its = [c for (s, sid, t), c in items.items() if t == k and all(f"ad_{j}_a1" in c for j in LABELS)]
            mat[k] = {j: float(np.mean([c[f"ad_{j}_a1"][metric] - c["base"][metric] for c in its])) for j in LABELS}
            w = [min(LABELS, key=lambda j: c[f"ad_{j}_a1"][metric]) for c in its]
            wins[k] = {j: w.count(j) for j in LABELS} | {"n": len(its)}
        conf[metric] = {"mean_change_vs_base": mat, "closest_label_counts": wins}
    return res, conf


def fry_pitch(items, gt_dir):
    """Section 6.6, 'Pitch': the vocal-fry control at half strength, and whether its D_dyn gain depends on the F0
    departure at full strength (F0 deviation per output from conditions.json)."""
    k, half, full = "vocal_fry", "ad_vocal_fry_a0.5", "ad_vocal_fry_a1"
    wrong_half = [f"ad_{j}_a0.5" for j in LABELS if j != k]
    out = {"a0.5_minus_base_dyn": boot(per_singer(items, k, lambda c: c[half]["d_dyn"] - c["base"]["d_dyn"]
                                                  if half in c else None)),
           "a0.5_correct_minus_wrong_dyn": boot(per_singer(items, k, lambda c: (c[half]["d_dyn"] - c["base"]["d_dyn"])
                                                           - np.mean([c[w]["d_dyn"] - c["base"]["d_dyn"] for w in wrong_half])
                                                           if half in c and all(w in c for w in wrong_half) else None))}
    cond = json.loads((gt_dir / "conditions.json").read_text(encoding="utf-8"))
    dev = {(c["singer"], c["sid"]): c["f0_mae_cents"] for c in cond
           if c.get("condition") == full and c.get("f0_mae_cents") is not None}
    pairs = [(dev[(s, sid)], conds[full]["d_dyn"] - conds["base"]["d_dyn"]) for (s, sid, t), conds in items.items()
             if t == k and full in conds and (s, sid) in dev]
    x, y = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731  (no ties in these data)
    rho = float(np.corrcoef(rank(x), rank(y))[0, 1])
    try:
        from scipy.stats import spearmanr
        p = float(spearmanr(x, y).pvalue)
    except ImportError:
        p = None
    out["spearman_f0_deviation_vs_change_in_dyn"] = {"rho": rho, "p": p, "n": len(pairs)}
    out["mean_dyn_reduction_by_f0_deviation"] = {"below_50_cents": {"n": int((x < 50).sum()), "reduction": float(-y[x < 50].mean())},
                                                 "50_cents_or_more": {"n": int((x >= 50).sum()), "reduction": float(-y[x >= 50].mean())}}
    return out


def classifier(cls_dir):
    summ = json.loads((cls_dir / "summary.json").read_text(encoding="utf-8"))
    preds = [json.loads(line) for line in (cls_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    out = {}
    for fs in ("full", "dyn"):
        s = summ[fs]
        o = {"heldout_balanced_accuracy": s["heldout_balanced_accuracy"], "l2": s["l2"],
             "loso_balanced_accuracy": s["loso_balanced_accuracy"][str(s["l2"])] if str(s["l2"]) in s["loso_balanced_accuracy"]
             else s["loso_balanced_accuracy"].get(s["l2"]), "heldout_confusion": s["heldout_confusion"]}
        eq = s.get("applied", {}).get("real_straight_oracleeq", {})
        o["real_plain_eq_to_technique"] = {k: eq.get(f"real_straight_oracleeq|{k}|", {}).get("hit_rate") for k in LABELS}
        o["real_plain_classified_as_straight"] = eq.get("base|None|", {}).get("predicted", {})
        part = [p for p in preds if p["feature_set"] == fs and p["part"] == "exp06a"]
        base = {(p["singer"], p["sid"]): p["probs"] for p in part if p["method"] == "base"}
        cond = {}
        for k in LABELS:
            for name, test in (("adapter_a1", lambda p: p["method"] == "adapter" and p["label"] == k and p["strength"] == 1.0),
                               ("adapter_a0.5", lambda p: p["method"] == "adapter" and p["label"] == k and p["strength"] == 0.5),
                               ("eqmatch", lambda p: p["method"] == "eqmatch" and p["label"] == k),
                               ("oracleeq", lambda p: p["method"] == "oracleeq" and p["label"] == k),
                               ("guidance", lambda p: p["method"] == "guidance" and p["label"] == k),
                               ("carried", lambda p: p["method"] == "carried" and p["label"] == k)):
                sel = [p for p in part if test(p)]
                if not sel:
                    continue
                g = {}
                for p in sel:
                    if (p["singer"], p["sid"]) in base:
                        g.setdefault(p["singer"], []).append(p["probs"][k] - base[p["singer"], p["sid"]][k])
                cond.setdefault(k, {})[name] = {
                    "hit_rate": float(np.mean([max(p["probs"], key=p["probs"].get) == k for p in sel])),
                    "p_label": float(np.mean([p["probs"][k] for p in sel])),
                    "p_minus_base": boot(g) if g else None, "n": len(sel)}
        o["exp06a"] = cond
        o["base_predicted"] = {c: sum(max(p, key=p.get) == c for p in base.values())
                               for c in ("straight",) + LABELS}
        early = s.get("applied", {}).get("earlier_cells", {})
        o["earlier_cells"] = {g: {k2: v for k2, v in st.items() if k2 in ("n", "hit_rate", "p_label", "p_label_minus_base",
                                                                        "pairs_increased", "pairs", "predicted")}
                              for g, st in early.items()}
        out[fs] = o
    return out


def report(gt, conf, cls):
    L = ["# Experiment 06 analysis", "",
         "Distances in dB (RMS per mel band of the cepstrally smoothed log-mel difference, level removed). "
         "Brackets: 95% cluster-bootstrap interval over the five held-out singers.", ""]
    L += ["## Mean distance to the real technique recording", "",
          "| Technique | n | Plain recording | Base | Adapter α=0.5 | Adapter α=1 | Matched EQ | Oracle EQ | Static bound | Guidance* | Technique in source |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in LABELS:
        m = gt[k]["means"]
        cell = lambda name: fmt(m[name]["d_raw"][0]) if m.get(name, {}).get("d_raw") else "–"  # noqa: E731
        L.append(f"| {NAMES[k]} | {gt[k]['n_items']} | {cell('real_straight')} | {cell('base')} | {cell('adapter_a0.5')} | "
                 f"{cell('adapter_a1')} | {cell('eqmatch')} | {cell('oracleeq')} | {cell('static_bound')} | "
                 f"{cell('guidance')} | {cell('carried')} |")
    L += ["", "*Guidance: scales on 'a' only (one exercise per singer).", "",
          "## Static and dynamic parts (D² = D_static² + D_dyn²)", "",
          "| Technique | Base static | Base dyn | Adapter static | Adapter dyn | Adapter − base, dyn | Correct − wrong labels, dyn | Static share of drop in D² |",
          "|---|---|---|---|---|---|---|---|"]
    for k in LABELS:
        m = gt[k]["means"]
        L.append(f"| {NAMES[k]} | {fmt(m['base']['d_static'][0])} | {fmt(m['base']['d_dyn'][0])} | "
                 f"{fmt(m['adapter_a1']['d_static'][0])} | {fmt(m['adapter_a1']['d_dyn'][0])} | "
                 f"{fmt(gt[k]['adapter_minus_base']['d_dyn'])} | {fmt(gt[k]['correct_minus_wrong']['d_dyn'])} | "
                 f"{fmt(gt[k]['static_share_of_drop'])} |")
    L += ["", "## Paired comparisons of the adapter (α = 1), raw distance", "",
          "| Technique | − base | − matched EQ | − oracle EQ | − static bound | Correct − wrong labels | Gap closed toward source-carried |",
          "|---|---|---|---|---|---|---|"]
    for k in LABELS:
        g = gt[k]
        L.append(f"| {NAMES[k]} | {fmt(g['adapter_minus_base']['d_raw'])} | {fmt(g['adapter_minus_eqmatch_raw'])} | "
                 f"{fmt(g['adapter_minus_oracleeq_raw'])} | {fmt(g['adapter_raw_minus_static_bound'])} | "
                 f"{fmt(g['correct_minus_wrong']['d_raw'])} | {fmt(g['gap_closed_toward_carried'])} |")
    L += ["", "## Which label brings the output closest to each real technique (counts over items)", ""]
    for metric in ("d_raw", "d_dyn"):
        L += [f"**{metric}**", "", "| Real technique \\ label | " + " | ".join(NAMES[j] for j in LABELS) + " | n |",
              "|---" * (len(LABELS) + 2) + "|"]
        for k in LABELS:
            w = conf[metric]["closest_label_counts"][k]
            L.append(f"| {NAMES[k]} | " + " | ".join(str(w[j]) for j in LABELS) + f" | {w['n']} |")
        L.append("")
    L += ["## Pre-specified checks", ""]
    for k in ("breathy", "vocal_fry"):
        g = gt[k]
        a = g["adapter_minus_base"]["d_dyn"]
        c = g["correct_minus_wrong"]["d_dyn"]
        L.append(f"- O1 {NAMES[k]}: adapter − base D_dyn {fmt(a)} ({'below zero' if a[2] < 0 else 'interval includes or exceeds zero'}); "
                 f"correct − wrong {fmt(c)} ({'below zero' if c[2] < 0 else 'interval includes or exceeds zero'}).")
        if cls:
            d = cls["dyn"]["exp06a"].get(k, {}).get("adapter_a1", {})
            if d.get("p_minus_base"):
                p = d["p_minus_base"]
                L.append(f"  dyn classifier P({NAMES[k]}) − base {fmt(p)} ({'above zero' if p[1] > 0 else 'interval includes or is below zero'}), hit rate {d['hit_rate']:.2f}.")
    for k in ("belt", "forte", "piano"):
        s = gt[k]["static_share_of_drop"]
        L.append(f"- O2 {NAMES[k]}: static share of the drop in D² {fmt(s)} (mean drop in D² {gt[k]['mean_drop_raw_sq']:+.2f}).")
    if cls:
        L += ["", "## Classifier", "", "| Feature set | LOSO bal. acc. | Held-out bal. acc. | Plain + oracle EQ classified as the technique (breathy / belt / fry / forte / piano) |",
              "|---|---|---|---|"]
        for fs in ("full", "dyn"):
            o = cls[fs]
            eq = " / ".join(fmt(o["real_plain_eq_to_technique"][k]) for k in LABELS)
            L.append(f"| {fs} | {fmt(o['loso_balanced_accuracy'])} | {fmt(o['heldout_balanced_accuracy'])} | {eq} |")
        for fs in ("full", "dyn"):
            L += ["", f"**{fs}: probability of the intended technique minus that of the base output (held-out singers)**", "",
                  "| Technique | Adapter α=0.5 | Adapter α=1 | Matched EQ | Oracle EQ | Guidance | Technique in source |",
                  "|---|---|---|---|---|---|---|"]
            for k in LABELS:
                d = cls[fs]["exp06a"].get(k, {})
                cell = lambda n: (fmt(d[n]["p_minus_base"]) + f" (hit {d[n]['hit_rate']:.2f})") if d.get(n, {}).get("p_minus_base") else "–"  # noqa: E731
                L.append(f"| {NAMES[k]} | {cell('adapter_a0.5')} | {cell('adapter_a1')} | {cell('eqmatch')} | "
                         f"{cell('oracleeq')} | {cell('guidance')} | {cell('carried')} |")
            L.append("")
            L.append(f"Base outputs classified as: {cls[fs]['base_predicted']}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--exact", action="store_true", help="enumerate all resamples instead of 2000 random ones")
    args = ap.parse_args(argv)
    global EXACT
    EXACT = args.exact
    root = Path(args.root)
    gt_dir, cls_dir = root / "results/exp06/groundtruth", root / "results/exp06/classifier"
    out = root / ("results/exp06/analysis_exact" if EXACT else "results/exp06/analysis")
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(gt_dir)
    gt, conf = groundtruth(items)
    cls = classifier(cls_dir) if (cls_dir / "summary.json").exists() else None
    pitch = fry_pitch(items, gt_dir)
    (out / "analysis.json").write_text(json.dumps({"groundtruth": gt, "confusion": conf, "classifier": cls,
                                                   "fry_pitch": pitch}, indent=1), encoding="utf-8")
    deltas = {f"{k}|{arm}|{m}": list(v) for k in LABELS for arm, ms in gt[k]["change_from_base"].items()
              for m, v in ms.items()}
    (out / "deltas.json").write_text(json.dumps(deltas, indent=1), encoding="utf-8")
    text = report(gt, conf, cls)
    text += ("\n## Adapter minus its matched equaliser (Table 7, column 2 is D_dyn)\n\n"
             "| Technique | D | D_static | D_dyn |\n|---|---|---|---|\n"
             + "".join(f"| {NAMES[k]} | " + " | ".join(fmt(gt[k]['adapter_minus_eqmatch'][m]) for m in ('d_raw', 'd_static', 'd_dyn'))
                       + " |\n" for k in LABELS))
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
