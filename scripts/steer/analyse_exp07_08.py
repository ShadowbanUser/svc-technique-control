"""Analysis of experiments 07 (conversion between held-out singers, Table 9) and 08 (GTSinger lyric phrases,
Section 6.8), plus the speaker-similarity and vocal-fry pitch costs of Section 6.6. NumPy only.

Run from anywhere: python scripts/steer/analyse_exp07_08.py
Reads results/exp07/{crosssinger/metrics.json, crosssinger/conditions.json, costs_similarity.json, costs_fry_pitch.json}
and results/exp08/{metrics.json, conditions.json}; prints the numbers and writes them to results/exp07_08_report.txt.
Intervals: cluster bootstrap, 2000 resamples per quantity from one generator seeded 20261119 (so the order of the
computations matters); clusters are the five target singers in experiment 07 and the 34 singer-song combinations in
experiment 08.
"""
from pathlib import Path
import json, numpy as np, collections
U = str(Path(__file__).resolve().parents[2] / "results")
LOG = []
_print = print


def print(*args, **kw):  # noqa: A001  (also keep a copy for the report file)
    _print(*args, **kw)
    LOG.append(" ".join(str(a) for a in args))

LABELS = ["breathy", "belt", "vocal_fry", "forte", "piano"]
rng = np.random.default_rng(20261119)

def boot(deltas, B=2000):
    """deltas: list of (cluster, value). Mean over items; clusters are drawn with replacement and all their items kept."""
    by = collections.defaultdict(list)
    for c, v in deltas: by[c].append(v)
    keys = sorted(by); arr = [np.array(by[k]) for k in keys]
    mean = float(np.mean([v for _, v in deltas]))
    bs = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        vals = np.concatenate([arr[i] for i in pick]); bs.append(vals.mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}] n={len(deltas)} k={len(keys)}"

def index(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        if r.get("d_raw") is None: continue
        by[(r["singer"], r["sid"], r["truth"])][r["condition"]] = r
    return by

print("=== EXP07 cross-singer (clusters = target singer) ===")
m7 = json.load(open(f"{U}/exp07/crosssinger/metrics.json"))
by = index(m7)
for t in LABELS:
    d1, d2, d3, d4, d5 = [], [], [], [], []
    wins = collections.Counter(); n = 0
    for (s, sid, tt), c in by.items():
        if tt != t or "base" not in c or f"ad_{t}_a1" not in c: continue
        a = c[f"ad_{t}_a1"]
        d1.append((s, a["d_raw"] - c["base"]["d_raw"]))
        d2.append((s, a["d_cmn"] - c[f"eqmatch_{t}"]["d_cmn"]))
        wrong = [c[f"ad_{j}_a1"] for j in LABELS if j != t]
        d3.append((s, a["d_cmn"] - np.mean([w["d_cmn"] for w in wrong])))
        d4.append((s, a["d_cmn"] - c["base"]["d_cmn"]))
        d5.append((s, c[f"oracleeq_{t}"]["d_raw"] - a["d_raw"]))
        wins[min(LABELS, key=lambda j: c[f"ad_{j}_a1"]["d_raw"])] += 1; n += 1
    print(f"{t:9s} raw ad-base {boot(d1)} | dyn ad-eqmatch {boot(d2)} | dyn ad-wrong {boot(d3)} | dyn ad-base {boot(d4)} | oracle-ad raw {boot(d5)} | closest {wins[t]}/{n}={wins[t]/n:.0%}")

print("=== EXP08 lyric phrases (clusters = song) ===")
m8 = json.load(open(f"{U}/exp08/metrics.json"))
c8 = json.load(open(f"{U}/exp08/conditions.json"))
by = index(m8)
def cl(s, sid, c): return s + "|" + c["base"]["real_file"].split("/")[-3]
comps = {"raw a1-base": ("ad_breathy_a1", "base", "d_raw"), "raw a0.5-base": ("ad_breathy_a0.5", "base", "d_raw"),
         "raw eqmatch-base": ("eqmatch_breathy", "base", "d_raw"), "raw oracle-base": ("oracleeq_breathy", "base", "d_raw"),
         "raw carried-base": ("carried_breathy", "base", "d_raw"), "raw realplain-base": ("real_straight", "base", "d_raw"),
         "dyn a1-base": ("ad_breathy_a1", "base", "d_cmn"), "dyn a0.5-base": ("ad_breathy_a0.5", "base", "d_cmn"),
         "dyn a1-eqmatch": ("ad_breathy_a1", "eqmatch_breathy", "d_cmn"), "dyn carried-base": ("carried_breathy", "base", "d_cmn"),
         "dyn realplain-base": ("real_straight", "base", "d_cmn"),
         "cpp-truth base": ("base", None, "cpp_db_minus_truth"), "cpp-truth a1": ("ad_breathy_a1", None, "cpp_db_minus_truth"),
         "cpp-truth carried": ("carried_breathy", None, "cpp_db_minus_truth"), "cpp-truth realplain": ("real_straight", None, "cpp_db_minus_truth"),
         "hf-truth base": ("base", None, "hf_ratio_db_minus_truth"), "hf-truth a1": ("ad_breathy_a1", None, "hf_ratio_db_minus_truth")}
for name, (a, b, k) in comps.items():
    d = [(cl(s, sid, c), c[a][k] - (c[b][k] if b else 0)) for (s, sid, t), c in by.items() if a in c and (b is None or b in c) and c[a].get(k) is not None]
    print(f"{name:20s} {boot(d)}")
d = []
for (s, sid, t), c in by.items():
    wrong = [c[f"ad_{j}_a1"] for j in LABELS if j != "breathy"]
    d.append((cl(s, sid, c), c["ad_breathy_a1"]["d_raw"] - np.mean([w["d_raw"] for w in wrong])))
print(f"{'raw a1-wrong':20s} {boot(d)}")
d = [(cl(s, sid, c), c["ad_breathy_a1"]["d_cmn"] - np.mean([c[f'ad_{j}_a1']['d_cmn'] for j in LABELS if j != 'breathy'])) for (s, sid, t), c in by.items()]
print(f"{'dyn a1-wrong':20s} {boot(d)}")
per = collections.defaultdict(list)
for (s, sid, t), c in by.items(): per[s].append(c["ad_breathy_a1"]["d_raw"] - c["base"]["d_raw"])
print("per singer raw a1-base", {k: round(float(np.mean(v)), 2) for k, v in per.items()})
cond = collections.defaultdict(list)
for r in c8:
    for k in ("similarity", "mean_abs_cents", "cents_mae", "f0_mae_cents"):
        if r.get(k) is not None: cond[(r["condition"], k)].append(r[k])
print({f"{a}|{k}": round(float(np.mean(v)), 3) for (a, k), v in sorted(cond.items())})
print("cond keys sample", [k for k in c8[1].keys()][:40])

print("=== COSTS ===")
sim = json.load(open(f"{U}/exp07/costs_similarity.json"))
g = collections.defaultdict(list)
for r in sim: g[(r["method"], r.get("label"), r.get("strength"))].append(r["similarity"])
for k in sorted(g, key=str): print(k, round(float(np.mean(g[k])), 3), len(g[k]))
fp = json.load(open(f"{U}/exp07/costs_fry_pitch.json"))
for k in ("real_dev_cents", "out_a1_dev_cents", "out_a0.5_dev_cents", "real_voiced_share", "out_a1_voiced_share"):
    v = [r[k] for r in fp if r.get(k) is not None]; print(k, round(float(np.mean(v)), 1), round(float(np.median(v)), 1), len(v))
c7 = json.load(open(f"{U}/exp07/crosssinger/conditions.json"))
g = collections.defaultdict(list)
for r in c7:
    if r.get("similarity") is not None: g[(r["method"], r.get("label"))].append(r["similarity"])
print("exp07 similarity", {str(k): round(float(np.mean(v)), 3) for k, v in g.items()})


print("=== EXP08 descriptors: change of the breathy control (alpha = 1) relative to the unmodified output ===")
cby = collections.defaultdict(dict)
for r in c8:
    cby[(r["singer"], r["sid"])][r["condition"]] = r
for name, key in (("cpp a1-base", "cpp_db"), ("hf a1-base", "hf_ratio_db"), ("cpp carried-base", "cpp_db")):
    cond = "carried_breathy" if "carried" in name else "ad_breathy_a1"
    d = [(s + "|" + c["base"]["song"], c[cond][key] - c["base"][key]) for (s, sid), c in cby.items()
         if cond in c and "base" in c and c[cond].get(key) is not None]
    print(f"{name:20s} {boot(d)}")

# Section 6.8, per singer (no resampling, so the intervals above are unaffected): change of the breathy control
# (alpha = 1) relative to the unmodified output, and the high-frequency ratio of each output minus that of the real
# breathy phrase
print("=== EXP08 per singer: breathy control (alpha = 1) minus unmodified output; HF ratio minus real breathy phrase ===")
m8by = {(s, sid): c for (s, sid, t), c in by.items()}
for singer in sorted({s for s, _ in cby}):
    keys = [k for k in cby if k[0] == singer and "ad_breathy_a1" in cby[k] and "base" in cby[k]]
    mk = [k for k in keys if k in m8by and "ad_breathy_a1" in m8by[k]]
    cpp = np.mean([cby[k]["ad_breathy_a1"]["cpp_db"] - cby[k]["base"]["cpp_db"] for k in keys])
    hf = np.mean([cby[k]["ad_breathy_a1"]["hf_ratio_db"] - cby[k]["base"]["hf_ratio_db"] for k in keys])
    raw = np.mean([m8by[k]["ad_breathy_a1"]["d_raw"] - m8by[k]["base"]["d_raw"] for k in mk])
    hfb = np.mean([m8by[k]["base"]["hf_ratio_db_minus_truth"] for k in mk])
    hfa = np.mean([m8by[k]["ad_breathy_a1"]["hf_ratio_db_minus_truth"] for k in mk])
    print(f"{singer:11s} n={len(keys)} | cpp a1-base {cpp:+.2f} | hf a1-base {hf:+.2f} | raw a1-base {raw:+.2f} | "
          f"hf-truth base {hfb:+.2f}, a1 {hfa:+.2f}")

with open(Path(U, "exp07_08_report.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(LOG) + "\n")
