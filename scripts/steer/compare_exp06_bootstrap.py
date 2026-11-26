"""Compare the intervals reported in the paper (2000 random cluster resamples, Section 5.6) with those from all
3125 ordered resamples of the five singers. Run after
    python scripts/steer/analyse_exp06.py
    python scripts/steer/analyse_exp06.py --exact
Prints the largest change of an interval end per table or figure, the cells that change by 0.015 or more, and any
interval whose inclusion of zero changes.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
rand = json.loads((ROOT / "results/exp06/analysis/analysis.json").read_text(encoding="utf-8"))
exact = json.loads((ROOT / "results/exp06/analysis_exact/analysis.json").read_text(encoding="utf-8"))

GROUPS = [("/adapter_minus_eqmatch/d_dyn", "Table 7, D_dyn adapter - matched EQ"),
          ("/adapter_minus_base/d_raw", "Table 7, D adapter - unmodified"),
          ("/correct_minus_wrong/", "Table 7, correct - wrong labels"),
          ("/static_share_of_drop", "Table 7, static share"),
          ("/adapter_minus_base/d_dyn", "Section 6.6 text, D_dyn adapter - unmodified"),
          ("fry_pitch", "Section 6.6, pitch"),
          ("/classifier/dyn/exp06a/", "Table 8 / Section 6.7, classifier")]


def group(path):
    if "/change_from_base/" in path and "/d_raw" not in path and any(a in path for a in ("/eqmatch/", "/oracleeq/", "/a1/")):
        return "Figure 3"
    if "/classifier/dyn/exp06a/" in path and "p_minus_base" not in path:
        return None
    for key, name in GROUPS:
        if key in path:
            return name
    return None


rows = []


def walk(a, b, path=""):
    if isinstance(a, dict):
        for k in a:
            walk(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and len(a) == 3 and all(isinstance(x, (int, float)) for x in a):
        rows.append((path, a, b))


walk(rand, exact)
worst, big, flips = defaultdict(float), [], []
for path, a, b in rows:
    if abs(a[0] - b[0]) > 1e-12:
        raise SystemExit(f"point estimates differ at {path}")
    if (a[1] <= 0 <= a[2]) != (b[1] <= 0 <= b[2]):
        flips.append((path, a, b))
    g = group(path)
    if g is None:
        continue
    d = max(abs(a[1] - b[1]), abs(a[2] - b[2]))
    worst[g] = max(worst[g], d)
    if d >= 0.015:
        big.append((g, path, a, b))
print("Largest change of an interval end (2000 resamples -> all 3125 resamples):")
for g, v in sorted(worst.items()):
    print(f"  {g:48s} {v:.3f}")
print("\nCells that change by 0.015 or more:")
for g, path, a, b in big:
    print(f"  {g}: {path}\n      2000: [{a[1]:+.3f}, {a[2]:+.3f}]   all: [{b[1]:+.3f}, {b[2]:+.3f}]")
print("\nIntervals whose inclusion of zero changes:", "none" if not flips else "")
for path, a, b in flips:
    print(f"  {path}: [{a[1]:+.4f}, {a[2]:+.4f}] -> [{b[1]:+.4f}, {b[2]:+.4f}]")
