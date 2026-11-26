"""Experiment 06B: an independent technique classifier, trained on real recordings only.

The classifier never sees converted audio during training and shares no weights with
Seed-VC or the adapter: hand-made spectral features (src/steer/objective.py) and a
multinomial logistic regression, six classes (straight, breathy, belt, vocal fry, forte,
piano). Training data: real VocalSet scales and arpeggios of the 15 training singers;
L2 strength chosen by leave-one-singer-out cross-validation.

Two feature sets:
  full  static spectral shape + how it changes over time
  dyn   only how it changes over time; a fixed equaliser cannot change these features

Checks and uses
  1. held-out singers, real recordings              does the classifier generalise?
  2. held-out straight takes equalised to the real   does it just read spectral tilt?
     technique take of the same exercise             (should stay 'straight' for dyn)
  3. experiment 06A outputs (held-out singers)       base / adapter / EQ / guidance / carried
  4. earlier test cells (exp03, exp04, exp05)        the listening-test material

Outputs: results/exp06/classifier/ (summary.json, predictions.jsonl, model.npz, features.pkl)
CPU only; no Seed-VC session is created.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/train"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prep_technique_pairs as pp  # noqa: E402
from exp03_panel import SR, EQ_LIMIT_DB, ltas_db, third_octave_smooth, apply_eq  # noqa: E402
from steer import objective as ob  # noqa: E402

OUT = ROOT / "results/exp06/classifier"
VOCALSET = ROOT / "data/vocalset_full/FULL"
GT = ROOT / "results/exp06/groundtruth"
HELD_OUT = ("female1", "female3", "female9", "male3", "male9")
SETS = ("scales", "arpeggios")
CLASSES = ("straight", "breathy", "belt", "vocal_fry", "forte", "piano")
FOLDER_CLASS = {"straight": "straight", **pp.TECHNIQUES}
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
FEATURE_SETS = ("full", "dyn")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def rel(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


def real_takes(singers):
    rows = []
    for singer in singers:
        for set_name in SETS:
            for folder, cls in FOLDER_CLASS.items():
                for vowel, files in sorted(pp.takes(VOCALSET / singer / set_name / folder).items()):
                    for f in files:
                        if "(" not in f.name:
                            rows.append({"singer": singer, "sid": f"{set_name}_{vowel}", "class": cls, "file": rel(f)})
    return rows


class FeatureCache:
    def __init__(self, path):
        self.path = path
        self.data = pickle.loads(path.read_bytes()) if path.exists() else {}
        self.new = 0

    def get(self, key, make_wave):
        if key not in self.data:
            self.data[key] = ob.technique_features(make_wave())
            self.new += 1
            if self.new % 200 == 0:
                self.save()
        return self.data[key]

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(self.data))
        os.replace(tmp, self.path)


def eq_to(wave, target, freqs):
    gain = third_octave_smooth(freqs, ltas_db(target) - ltas_db(wave))
    gain[freqs < 60] = 0.0
    return apply_eq(wave, np.clip(gain, -EQ_LIMIT_DB, EQ_LIMIT_DB))


class Model:
    def __init__(self, x, y, singers, l2_grid=L2_GRID):
        k = len(CLASSES)
        counts = np.bincount(y, minlength=k)
        self.class_weight = len(y) / (k * np.maximum(counts, 1))
        cv = {}
        for l2 in l2_grid:
            scores = []
            for held in sorted(set(singers)):
                tr, te = singers != held, singers == held
                mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
                W, b = ob.fit_logreg((x[tr] - mu) / sd, y[tr], k, l2, self.class_weight[y[tr]])
                pred = ob.predict_proba((x[te] - mu) / sd, W, b).argmax(1)
                scores.append(ob.balanced_accuracy(y[te], pred, k))
            cv[l2] = float(np.mean([s for s in scores if s is not None]))
        self.cv = cv
        self.l2 = max(cv, key=cv.get)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-8
        self.W, self.b = ob.fit_logreg((x - self.mu) / self.sd, y, k, self.l2, self.class_weight[y])

    def proba(self, x):
        return ob.predict_proba((np.atleast_2d(x) - self.mu) / self.sd, self.W, self.b)


def confusion(y, pred):
    k = len(CLASSES)
    m = np.zeros((k, k), dtype=int)
    for a, b in zip(y, pred):
        m[a, b] += 1
    return {CLASSES[i]: {CLASSES[j]: int(m[i, j]) for j in range(k)} for i in range(k)}


def parse_earlier():
    """Earlier cross-singer test cells (6 s excerpts, targets f3/f9). Returns condition records."""
    records = []
    techs = "breathy|belt|vocal_fry|forte|piano"
    for exp, folder in (("exp04", "results/exp04/eval/wav"), ("exp05", "results/exp05/sweep/wav")):
        for path in sorted((ROOT / folder).glob("*.wav")):
            m = re.match(r"^(f3|f9)_(row|dona)_(.+)\.wav$", path.name)
            if not m:
                continue
            cell, arm = f"{m.group(1)}_{m.group(2)}", m.group(3)
            if arm == "base":
                info = {"method": "base", "label": None}
            elif re.match(rf"^({techs})_a[\d.]+$", arm):
                tech, amount = arm.rsplit("_a", 1)
                info = {"method": "adapter", "label": tech, "strength": float(amount)}
            elif re.match(rf"^({techs})_eqmatch_a1$", arm):
                info = {"method": "eqmatch", "label": arm.split("_eqmatch")[0]}
            else:
                continue
            records.append({"experiment": exp, "cell": cell, "condition": arm, "file": rel(path), **info})
    axis_label = {"breathy": "breathy", "belt": "belt", "fry": "vocal_fry", "forte": "forte"}
    for path in sorted((ROOT / "results/exp03/wav").glob("*.wav")):
        m = re.match(r"^(f3|f9)_(row|dona)_(.+)\.wav$", path.name)
        if not m:
            continue
        cell, arm = f"{m.group(1)}_{m.group(2)}", m.group(3)
        if arm == "base":
            records.append({"experiment": "exp03", "cell": cell, "condition": arm, "file": rel(path),
                            "method": "base", "label": None})
            continue
        axis, _, rest = arm.partition("_")
        if axis not in axis_label:
            continue
        info = None
        if rest == "cgm_w4":
            info = {"method": "guidance", "label": axis_label[axis]}
        elif rest == "cgm_w-4" and axis == "forte":
            info = {"method": "guidance", "label": "piano"}
        elif rest == "eqmatch_cgm_w4":
            info = {"method": "eqmatch_guidance", "label": axis_label[axis]}
        elif rest == "eq_a1":
            info = {"method": "donor_eq", "label": axis_label[axis]}
        if info:
            records.append({"experiment": "exp03", "cell": cell, "condition": arm, "file": rel(path), **info})
    return records


def summarise_group(records, probs, key_fn):
    """records/probs aligned; returns {group: stats}; stats per label: P(label), hit rate, shift vs base."""
    base_p = {}
    for r, p in zip(records, probs):
        if r["method"] == "base":
            base_p[r["pair"]] = p
    groups = {}
    for r, p in zip(records, probs):
        groups.setdefault(key_fn(r), []).append((r, p))
    out = {}
    for g, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        stats = {"n": len(members),
                 "predicted": {c: int(sum(int(np.argmax(p)) == i for _, p in members)) for i, c in enumerate(CLASSES)},
                 "mean_p": {c: float(np.mean([p[i] for _, p in members])) for i, c in enumerate(CLASSES)}}
        label = members[0][0].get("label")
        if label in CLASSES:
            i = CLASSES.index(label)
            stats["p_label"] = float(np.mean([p[i] for _, p in members]))
            stats["hit_rate"] = float(np.mean([int(np.argmax(p)) == i for _, p in members]))
            shifts = [p[i] - base_p[r["pair"]][i] for r, p in members if r["pair"] in base_p]
            if shifts:
                stats["p_label_minus_base"] = float(np.mean(shifts))
                stats["pairs_increased"] = int(sum(s > 0 for s in shifts))
                stats["pairs"] = len(shifts)
        out[str(g)] = stats
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "summary.json").exists():
        log(f"{OUT} already has summary.json; nothing to do")
        return 0
    if (OUT / "failure.json").exists():
        os.replace(OUT / "failure.json", OUT / f"failure_{int(time.time())}.json")
    cache = FeatureCache(OUT / "features.pkl")
    singers_all = sorted(p.name for p in VOCALSET.iterdir() if p.is_dir())
    train_singers = [s for s in singers_all if s not in HELD_OUT]
    train_rows, test_rows = real_takes(train_singers), real_takes([s for s in HELD_OUT if s in singers_all])
    log(f"{len(train_rows)} training takes ({len(train_singers)} singers), {len(test_rows)} held-out takes")

    def feats(rows):
        out = []
        for n, r in enumerate(rows, 1):
            out.append(cache.get(r["file"], lambda r=r: pp.load_trimmed(ROOT / r["file"])))
            if n % 250 == 0:
                log(f"  features {n}/{len(rows)}")
        return out

    started = time.perf_counter()
    f_train, f_test = feats(train_rows), feats(test_rows)
    cache.save()
    log(f"real-recording features done in {time.perf_counter() - started:.0f}s")
    y_train = np.array([CLASSES.index(r["class"]) for r in train_rows])
    y_test = np.array([CLASSES.index(r["class"]) for r in test_rows])
    groups = np.array([r["singer"] for r in train_rows])

    models, summary = {}, {"status": "completed", "classes": CLASSES, "train_singers": train_singers,
                           "held_out": HELD_OUT, "n_train": len(train_rows), "n_test": len(test_rows),
                           "feature_names": ob.feature_names()}
    for fs in FEATURE_SETS:
        x = np.stack([f[fs] for f in f_train])
        model = Model(x, y_train, groups)
        models[fs] = model
        pred = model.proba(np.stack([f[fs] for f in f_test])).argmax(1)
        summary[fs] = {"l2": model.l2, "loso_balanced_accuracy": model.cv,
                       "heldout_balanced_accuracy": ob.balanced_accuracy(y_test, pred, len(CLASSES)),
                       "heldout_accuracy": float(np.mean(pred == y_test)), "heldout_confusion": confusion(y_test, pred)}
        log(f"{fs}: l2 {model.l2:g}, LOSO bal. acc {model.cv[model.l2]:.3f}, "
            f"held-out bal. acc {summary[fs]['heldout_balanced_accuracy']:.3f}")
    np.savez(OUT / "model.npz", **{f"{fs}_{k}": getattr(models[fs], k) for fs in FEATURE_SETS
                                   for k in ("W", "b", "mu", "sd")})

    # ---- 2. held-out straight takes equalised to the real technique take -----------------
    freqs = np.fft.rfftfreq(2048, 1.0 / SR)
    straight = {(r["singer"], r["sid"]): r for r in test_rows if r["class"] == "straight"}
    eq_records, eq_feats = [], []
    for r in test_rows:
        s = straight.get((r["singer"], r["sid"]))
        if r["class"] == "straight" or s is None:
            continue
        key = f"eq::{s['file']}::{r['file']}"
        eq_feats.append(cache.get(key, lambda s=s, r=r: eq_to(pp.load_trimmed(ROOT / s["file"]),
                                                             pp.load_trimmed(ROOT / r["file"]), freqs)))
        eq_records.append({"method": "real_straight_oracleeq", "label": r["class"], "singer": r["singer"],
                           "sid": r["sid"], "pair": (r["singer"], r["sid"]), "file": r["file"]})
    # the unmodified straight takes serve as the 'base' for the shift statistic
    for (singer, sid), s in straight.items():
        eq_records.append({"method": "base", "label": None, "singer": singer, "sid": sid, "pair": (singer, sid),
                           "file": s["file"]})
        eq_feats.append(cache.get(s["file"], lambda s=s: pp.load_trimmed(ROOT / s["file"])))
    cache.save()

    # ---- 3. experiment 06A outputs ------------------------------------------------------
    gt_records, gt_feats = [], []
    conditions_path = GT / "conditions.json"
    if conditions_path.exists():
        for c in json.loads(conditions_path.read_text(encoding="utf-8")):
            if c["method"] == "real":
                continue
            gt_records.append({**{k: c.get(k) for k in ("singer", "sid", "condition", "method", "label", "strength",
                                                       "file")}, "pair": (c["singer"], c["sid"])})
            gt_feats.append(cache.get(c["file"], lambda c=c: pp.load_trimmed(ROOT / c["file"])))
        cache.save()
        log(f"experiment 06A outputs classified: {len(gt_records)}")
    else:
        log("no experiment 06A outputs found; skipping that part")

    # ---- 4. earlier cross-singer cells --------------------------------------------------
    early_records = parse_earlier()
    early_feats = []
    for r in early_records:
        r["pair"] = (r["experiment"], r["cell"])
        early_feats.append(cache.get(r["file"], lambda r=r: pp.load_trimmed(ROOT / r["file"])))
    cache.save()
    log(f"earlier test cells classified: {len(early_records)}")

    def key_gt(r):
        return f"{r['method']}|{r['label']}|{r.get('strength') if r['method'] == 'adapter' else ''}"

    def key_early(r):
        return f"{r['experiment']}|{r['method']}|{r['label']}|{r.get('strength') if r['method'] == 'adapter' else ''}"

    with (OUT / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for fs in FEATURE_SETS:
            model = models[fs]
            block = {}
            for name, records, fl, key_fn in (("real_straight_oracleeq", eq_records, eq_feats, key_gt),
                                              ("exp06a", gt_records, gt_feats, key_gt),
                                              ("earlier_cells", early_records, early_feats, key_early)):
                if not records:
                    continue
                probs = model.proba(np.stack([f[fs] for f in fl]))
                block[name] = summarise_group(records, probs, key_fn)
                for r, p in zip(records, probs):
                    stream.write(json.dumps({"feature_set": fs, "part": name,
                                             **{k: v for k, v in r.items() if k != "pair"},
                                             "probs": dict(zip(CLASSES, map(float, p)))}) + "\n")
            summary[fs]["applied"] = block
    summary["minutes"] = (time.perf_counter() - started) / 60
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    for fs in FEATURE_SETS:
        applied = summary[fs].get("applied", {})
        for part in ("real_straight_oracleeq", "exp06a"):
            for g, st in applied.get(part, {}).items():
                if "p_label" in st and (g.startswith("adapter") and "|1.0" in g or not g.startswith("adapter")):
                    log(f"{fs:4s} {part:22s} {g:28s} hit {st['hit_rate']:.2f}  P(label) {st['p_label']:.2f}  "
                        f"vs base {st.get('p_label_minus_base', float('nan')):+.2f}")
    return 0


if __name__ == "__main__":
    t0 = time.perf_counter()
    try:
        code = main()
    except Exception as error:
        traceback.print_exc()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "failure.json").write_text(json.dumps({"error_type": type(error).__name__, "error": str(error)}),
                                          encoding="utf-8")
        code = 1
    log(f"total wall {time.perf_counter() - t0:.1f}s, exit {code}")
    raise SystemExit(code)
