"""Decode and analyse the multi-listener test (result strings SVCLT1|...).

Usage
  python analyse_listening.py --key results/listening_main/key.json \
      --responses results/listening_main/responses.txt [--audio-dir listening_test/audio] [--out DIR]

responses.txt: one result string per line, as copied from the test page (blank lines and
lines not starting with SVCLT1 are ignored; a listener who submitted twice is counted once,
the last submission wins).

Pre-specified analysis (preregistration, Analysis; paper Section 5.7):
  identification  per condition x technique: proportion naming the intended change; full
                  response distribution; listener-level test that the intended response is
                  chosen more often than the most frequent alternative (one-sided Wilcoxon
                  signed-rank over listeners, Holm over the five techniques of a condition);
                  Bayesian mixed-effects logistic regression with crossed random intercepts
                  for listeners and stimuli as a secondary estimate (if statsmodels exists)
  matched EQ      proportion choosing the controlled output as more like a change of technique
                  among {controlled, equaliser}; same listener-level test against the equaliser
  screening       listeners who rate both identical calibration pairs as clearly different
  H3              Spearman correlation across stimuli between descriptor change and
                  identification rate (needs --audio-dir)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TECHS = ("breathy", "belt", "vocal_fry", "forte", "piano")
QI2 = ["no change", "post-processing", "breathier", "brighter/forceful", "rough/creaky", "fuller/stronger",
       "softer/lighter", "other technique", "different singer"]
INTENDED = {"breathy": 3, "belt": 4, "vocal_fry": 5, "forte": 6, "piano": 7}
DEMO_KEYS = ("age", "grp", "mus", "svs", "out")
GROUPS = {"1": "computing", "2": "creator", "3": "both", "4": "other"}


def parse(line):
    parts = line.strip().split("|")
    if len(parts) < 7 or parts[0] != "SVCLT1":
        raise ValueError("not an SVCLT1 string")
    lid, demo = parts[1], parts[2]
    fields = {p.split(":", 1)[0]: p.split(":", 1)[1] for p in parts[3:] if ":" in p}
    d = dict(re.findall(r"(age|grp|mus|svs|out)([0-9?])", demo))
    ident, eq = {}, {}
    for item in filter(None, fields.get("I", "").split(",")):
        num, ans = item.split("-", 1)
        ident["I" + num] = ans
    for item in filter(None, fields.get("E", "").split(",")):
        head, ans = item.split("-", 1)
        num, swap = head.split("x")
        eq["E" + num] = (int(swap), ans)
    return {"lid": lid, "demo": d, "hp": fields.get("hp", ""), "I": ident, "E": eq, "note": fields.get("note", "")}


def digit(ans, i):
    c = ans[i] if i < len(ans) else "?"
    return int(c) if c.isdigit() else None


def decode(listeners, key):
    trials = {t["id"]: t for t in key["trials"]}
    rows = []
    for L in listeners:
        for tid, ans in L["I"].items():
            t = trials.get(tid)
            if t is None or t["type"] != "I":
                continue
            rows.append({"listener": L["lid"], "group": GROUPS.get(L["demo"].get("grp"), "?"), "trial": tid, "type": "I",
                         "condition": t["condition"], "technique": t["technique"], "strength": t.get("strength"),
                         "stimulus": t["files"]["S"], "noticeable": digit(ans, 0), "change": digit(ans, 1),
                         "quality": digit(ans, 2), "same_singer": digit(ans, 3), "would_use": digit(ans, 4)})
        for tid, (swap, ans) in L["E"].items():
            t = trials.get(tid)
            if t is None or t["type"] != "E":
                continue
            q2, q3 = digit(ans, 1), digit(ans, 2)
            # page shows A = neural, B = equaliser unless swapped
            to_side = {1: "neural" if not swap else "eq", 2: "eq" if not swap else "neural"}
            more_tech = {**to_side, 3: "both", 4: "neither", 5: "cannot tell"}.get(q2)
            use = {**to_side, 3: "either", 4: "neither"}.get(q3)
            qa, qb = digit(ans, 3), digit(ans, 4)
            rows.append({"listener": L["lid"], "group": GROUPS.get(L["demo"].get("grp"), "?"), "trial": tid, "type": "E",
                         "condition": t["condition"], "technique": t["technique"], "strength": t.get("strength"),
                         "stimulus": t["files"]["A"], "difference": digit(ans, 0), "more_technique": more_tech,
                         "would_use": use, "quality_neural": qa if not swap else qb,
                         "quality_eq": qb if not swap else qa, "catch": bool(t.get("catch"))})
    return rows


def screen(rows, listeners):
    flagged = {}
    for L in listeners:
        catch = [r for r in rows if r["listener"] == L["lid"] and r["type"] == "E" and r["catch"]]
        answered = len(L["I"]) + len(L["E"])
        missing = sum(a.count("?") for a in L["I"].values()) + sum(a.count("?") for _, a in L["E"].values())
        reason = []
        if catch and len(catch) >= 2 and all(r["difference"] == 3 for r in catch):
            reason.append("both calibration pairs rated clearly different")
        flagged[L["lid"]] = {"excluded": bool(reason), "reason": reason, "trials": answered, "missing_answers": missing}
    return flagged


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [None] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def wilcoxon_greater(x):
    from scipy.stats import wilcoxon
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) < 2 or np.allclose(x, 0):
        return None
    try:
        return float(wilcoxon(x, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        return None


def bayes_glmm(rows_binary):
    """rows_binary: list of (y, listener, stimulus). Returns posterior mean and 95% interval of the intercept (logit)."""
    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception:
        return None
    if len(rows_binary) < 10 or len({y for y, _, _ in rows_binary}) < 2:
        return None
    df = pd.DataFrame(rows_binary, columns=["y", "listener", "stimulus"])
    model = BinomialBayesMixedGLM.from_formula("y ~ 1", {"listener": "0 + C(listener)", "stimulus": "0 + C(stimulus)"}, df)
    fit = model.fit_vb()
    m, s = float(fit.fe_mean[0]), float(fit.fe_sd[0])
    return {"logit_mean": m, "logit_lo": m - 1.96 * s, "logit_hi": m + 1.96 * s}


def identification(rows):
    out = {}
    by = defaultdict(list)
    for r in rows:
        if r["type"] == "I" and r["change"] is not None:
            by[r["condition"], r["technique"]].append(r)
    for (cond, tech), rs in sorted(by.items()):
        k = INTENDED[tech]
        counts = Counter(r["change"] for r in rs)
        alt = max((o for o in range(1, 10) if o != k), key=lambda o: (counts.get(o, 0), -o))
        per_l = defaultdict(list)
        for r in rs:
            per_l[r["listener"]].append((r["change"] == k) - (r["change"] == alt))
        scores = [float(np.mean(v)) for v in per_l.values()]
        out[f"{cond}|{tech}"] = {
            "n": len(rs), "listeners": len(per_l), "intended": QI2[k - 1], "intended_rate": counts.get(k, 0) / len(rs),
            "most_frequent_alternative": QI2[alt - 1], "alternative_rate": counts.get(alt, 0) / len(rs),
            "distribution": {QI2[o - 1]: counts.get(o, 0) / len(rs) for o in range(1, 10)},
            "any_technique_rate": sum(counts.get(o, 0) for o in range(3, 9)) / len(rs),
            "post_processing_rate": counts.get(2, 0) / len(rs),
            "clear_distortion_rate": float(np.mean([r["quality"] == 3 for r in rs if r["quality"]])) if rs else None,
            "same_singer_rate": float(np.mean([r["same_singer"] == 1 for r in rs if r["same_singer"]])) if rs else None,
            "usable_rate": float(np.mean([r["would_use"] in (1, 2) for r in rs if r["would_use"]])) if rs else None,
            "p_intended_vs_alternative": wilcoxon_greater(scores),
            "glmm": bayes_glmm([(int(r["change"] == k), r["listener"], r["stimulus"]) for r in rs if r["change"] in (k, alt)]),
        }
    return out


def matched_eq(rows):
    out = {}
    by = defaultdict(list)
    for r in rows:
        if r["type"] == "E" and not r["catch"] and r["more_technique"] is not None:
            by[r["condition"], r["technique"]].append(r)
    for (cond, tech), rs in sorted(by.items()):
        c = Counter(r["more_technique"] for r in rs)
        decided = c["neural"] + c["eq"]
        per_l = defaultdict(list)
        for r in rs:
            per_l[r["listener"]].append((r["more_technique"] == "neural") - (r["more_technique"] == "eq"))
        uses = Counter(r["would_use"] for r in rs)
        out[f"{cond}|{tech}"] = {
            "n": len(rs), "listeners": len(per_l),
            "neural_share_of_decided": c["neural"] / decided if decided else None,
            "distribution": {k: c[k] / len(rs) for k in ("neural", "eq", "both", "neither", "cannot tell")},
            "no_difference_rate": float(np.mean([r["difference"] == 1 for r in rs if r["difference"]])),
            "would_use": {k: uses[k] / len(rs) for k in ("neural", "eq", "either", "neither")},
            "clear_distortion_neural": float(np.mean([r["quality_neural"] == 3 for r in rs if r["quality_neural"]])),
            "clear_distortion_eq": float(np.mean([r["quality_eq"] == 3 for r in rs if r["quality_eq"]])),
            "p_neural_vs_eq": wilcoxon_greater([float(np.mean(v)) for v in per_l.values()]),
            "glmm": bayes_glmm([(int(r["more_technique"] == "neural"), r["listener"], r["stimulus"]) for r in rs
                                if r["more_technique"] in ("neural", "eq")]),
        }
    return out


def add_holm(block, key):
    for cond in sorted({k.split("|")[0] for k in block}):
        names = [k for k in block if k.startswith(cond + "|") and block[k][key] is not None]
        for name, adj in zip(names, holm([block[n][key] for n in names])):
            block[name][key + "_holm"] = adj


def descriptors(rows, key, audio_dir):
    """Per identification stimulus: change of CPP and HF ratio relative to its base file."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import librosa
    from steer.metrics import summarize
    from scipy.stats import spearmanr
    cache = {}

    def desc(code):
        if code not in cache:
            w, _ = librosa.load(str(audio_dir / code), sr=16000)
            cache[code] = summarize(w)
        return cache[code]
    rate = defaultdict(list)
    for r in rows:
        if r["type"] == "I" and r["change"] is not None:
            rate[r["trial"]].append(r["change"] == INTENDED[r["technique"]])
    pts = []
    for t in key["trials"]:
        if t["type"] != "I" or t["id"] not in rate:
            continue
        s, b = desc(t["files"]["S"]), desc(t["files"]["base"])
        pts.append({"trial": t["id"], "condition": t["condition"], "technique": t["technique"],
                    "identification_rate": float(np.mean(rate[t["id"]])),
                    "d_cpp": s["cpp_db"] - b["cpp_db"], "d_hf": s["hf_ratio_db"] - b["hf_ratio_db"]})
    out = {"stimuli": pts}
    for m in ("d_cpp", "d_hf"):
        x = [abs(p[m]) for p in pts]
        y = [p["identification_rate"] for p in pts]
        if len(pts) > 3:
            rho, p = spearmanr(x, y)
            out[f"spearman_abs_{m}"] = {"rho": float(rho), "p": float(p), "n": len(pts)}
    return out


def report(summary):
    L = ["# Listening test analysis", "",
         f"Listeners: {summary['n_listeners']} submitted, {summary['n_included']} included "
         f"({summary['n_excluded']} excluded by the calibration screen). Groups: {summary['groups']}.", "",
         "## Identification", "",
         "| Condition | Technique | n | Intended | Rate | Most frequent alternative | Rate | Any technique | Post-processing | p (Holm) | Clear distortion | Usable |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in summary["identification"].items():
        cond, tech = k.split("|")
        p = v.get("p_intended_vs_alternative_holm")
        L.append(f"| {cond} | {tech} | {v['n']} | {v['intended']} | {v['intended_rate']:.2f} | {v['most_frequent_alternative']} | "
                 f"{v['alternative_rate']:.2f} | {v['any_technique_rate']:.2f} | {v['post_processing_rate']:.2f} | "
                 f"{'–' if p is None else f'{p:.3f}'} | {v['clear_distortion_rate']:.2f} | {v['usable_rate']:.2f} |")
    L += ["", "## Matched equaliser", "",
          "| Condition | Technique | n | Controlled chosen (of decided) | Equaliser | Both | Neither | Cannot tell | No difference heard | p (Holm) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in summary["matched_eq"].items():
        cond, tech = k.split("|")
        d = v["distribution"]
        p = v.get("p_neural_vs_eq_holm")
        share = "–" if v["neural_share_of_decided"] is None else f"{v['neural_share_of_decided']:.2f}"
        L.append(f"| {cond} | {tech} | {v['n']} | {share} | {d['eq']:.2f} | {d['both']:.2f} | {d['neither']:.2f} | "
                 f"{d['cannot tell']:.2f} | {v['no_difference_rate']:.2f} | {'–' if p is None else f'{p:.3f}'} |")
    if summary.get("h3"):
        L += ["", "## H3", "", json.dumps({k: v for k, v in summary["h3"].items() if k != "stimuli"})]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--audio-dir")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else Path(args.key).parent / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    listeners, errors = {}, []
    for n, line in enumerate(Path(args.responses).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line.startswith("SVCLT1"):
            continue
        try:
            L = parse(line)
            listeners[L["lid"]] = L
        except Exception as error:
            errors.append(f"line {n}: {error}")
    listeners = list(listeners.values())
    rows = decode(listeners, key)
    flags = screen(rows, listeners)
    keep = {lid for lid, f in flags.items() if not f["excluded"]}
    used = [r for r in rows if r["listener"] in keep]
    summary = {"n_listeners": len(listeners), "n_included": len(keep), "n_excluded": len(listeners) - len(keep),
               "parse_errors": errors, "screening": flags,
               "groups": dict(Counter(GROUPS.get(L["demo"].get("grp"), "?") for L in listeners if L["lid"] in keep)),
               "demographics": {k: dict(Counter(L["demo"].get(k, "?") for L in listeners if L["lid"] in keep)) for k in DEMO_KEYS},
               "identification": identification(used), "matched_eq": matched_eq(used)}
    add_holm(summary["identification"], "p_intended_vs_alternative")
    add_holm(summary["matched_eq"], "p_neural_vs_eq")
    by_group = {}
    for g in sorted({r["group"] for r in used}):
        sub = [r for r in used if r["group"] == g]
        by_group[g] = {"identification": {k: v["intended_rate"] for k, v in identification(sub).items()},
                       "matched_eq": {k: v["neural_share_of_decided"] for k, v in matched_eq(sub).items()}}
    summary["by_group"] = by_group
    if args.audio_dir:
        summary["h3"] = descriptors(used, key, Path(args.audio_dir))
    with (out / "responses_long.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in rows for k in r}) + ["excluded"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "excluded": r["listener"] not in keep})
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    text = report(summary)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
