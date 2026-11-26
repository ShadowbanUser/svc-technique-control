"""Experiment 08, phoneme-level follow-up (Section 6.8): is the breathy control closer to the real breathy phrase at
the positions that GTSinger labels as breathy?

GTSinger annotates every phoneme of a take with a breathy flag (<phrase>.json next to the audio). Each frame of the
real breathy take (100 frames/s) is assigned to one of two sets: sung phonemes labelled breathy, and sung phonemes
not labelled breathy; silences and breaths (<SP>, <AP>) are left out. Through the alignment path of experiment 08,
each output is compared with the real breathy take on each set separately, exactly as in experiment 08 but restricted
to the set: D and D_dyn (the mean is removed within the set) over the frames active in the real take, and CPP and the
high-frequency ratio of the output at the mapped frames (voiced frames as in steer.metrics.summarize). A set is used
for a phrase only if it has at least MIN_FRAMES compared frames.

Needs the outputs of experiment 08 (results/exp08/wav, results/exp08/align) and the GTSinger audio and annotation
files of the phrases in data/exp08_manifest.json (data/gtsinger/<path in the GTSinger repository>, .wav and .json).
Writes results/exp08/phonemes.json and results/exp08/phonemes_report.txt.

Intervals: cluster bootstrap over the 34 singer-song combinations (as in experiment 08), 2000 resamples,
numpy.random.default_rng(20261119) re-seeded for every quantity, so the order of the computations does not matter.
Check: with all active frames as the set, D and D_dyn equal results/exp08/metrics.json and CPP and the high-frequency
ratio equal results/exp08/conditions.json; the largest differences are printed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from steer import objective as ob  # noqa: E402
from steer.metrics import cpp_track, hf_ratio_track, summarize, _frames  # noqa: E402

TRUTH = "breathy"
MIN_FRAMES = 50          # 0.5 s
CPP_OFFSET = 2           # steer.metrics frames start at the frame (40 ms window); objective frames are centred
B, SEED = 2000, 20261119
CONDITIONS = {"base": "base", "a1": "ad_breathy_a1", "a0.5": "ad_breathy_a0.5", "eqmatch": "eqmatch_breathy"}
SETS = ("breathy", "other")


def load(path, sr):
    import librosa
    return librosa.load(str(path), sr=sr, mono=True)[0]


def frame_sets(annotation, n_frames):
    """Label of each real-take frame (time = frame / 100 s): 'breathy', 'other' or None (silence, breath)."""
    labels = np.full(n_frames, None, dtype=object)
    t = np.arange(n_frames) / 100.0
    for word in annotation:
        for ph, start, end, flag in zip(word["ph"], word["ph_start"], word["ph_end"], word["breathy"]):
            if ph in ("<SP>", "<AP>"):
                continue
            labels[(t >= start) & (t < end)] = "breathy" if str(flag) == "1" else "other"
    return labels


def descriptor_on(wave16, frames_100):
    """Mean CPP and high-frequency ratio of a 16 kHz signal at the given objective frames (repeats count)."""
    cpp, hf = cpp_track(wave16), hf_ratio_track(wave16)
    rms = np.sqrt((_frames(wave16) ** 2).mean(1))
    voiced = rms > 0.1 * np.nanmax(rms)
    i = np.clip(np.asarray(frames_100) - CPP_OFFSET, 0, len(cpp) - 1)
    keep = voiced[i] & np.isfinite(cpp[i])
    if keep.sum() < MIN_FRAMES:
        return None, None
    return float(np.mean(cpp[i][keep])), float(np.mean(hf[i][keep]))


def boot(by_cluster):
    keys = sorted(by_cluster)
    arr = [np.asarray(by_cluster[k], float) for k in keys]
    allv = np.concatenate(arr)
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([arr[i] for i in rng.integers(0, len(keys), len(keys))]).mean() for _ in range(B)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(allv.mean()), float(lo), float(hi), int(len(allv)), len(keys)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "results/exp08"), help="folder with wav/ and align/ of experiment 08")
    ap.add_argument("--data", default=str(ROOT / "data/gtsinger"), help="GTSinger audio (.wav) root")
    ap.add_argument("--annotations", default=None, help="GTSinger annotation (.json) root, default: --data")
    ap.add_argument("--out", default=str(ROOT / "results/exp08"))
    args = ap.parse_args(argv)
    outputs, data = Path(args.outputs), Path(args.data)
    annotations = Path(args.annotations) if args.annotations else data
    manifest = json.loads((ROOT / "data/exp08_manifest.json").read_text(encoding="utf-8"))
    metrics = {(r["singer"], r["sid"], r["condition"]): r for r in
               json.loads((ROOT / "results/exp08/metrics.json").read_text(encoding="utf-8")) if r.get("d_raw") is not None}
    conds = {(r["singer"], r["sid"], r["condition"]): r for r in
             json.loads((ROOT / "results/exp08/conditions.json").read_text(encoding="utf-8"))}

    rows, checksums, check = [], {}, collections.defaultdict(float)
    for singer, entry in manifest["singers"].items():
        for n, pair in enumerate(entry["pairs"]):
            sid, key = f"{n:02d}", f"{singer}_{n:02d}"
            ann_path = annotations / (pair["breathy"][:-4] + ".json")
            raw = ann_path.read_bytes()
            checksums[pair["breathy"][:-4] + ".json"] = hashlib.sha256(raw).hexdigest()
            real44 = load(data / pair["breathy"], ob.SR)
            real_an = ob.analyse(real44)
            labels = frame_sets(json.loads(raw.decode("utf-8")), real_an["frames"])
            mapping = json.loads((outputs / "align" / f"{key}.json").read_text(encoding="utf-8"))[TRUTH]["mapping50"]
            real16 = load(data / pair["breathy"], 16000)
            row = {"singer": singer, "sid": sid, "song": pair["song"], "cluster": f"{singer}|{pair['song']}",
                   "frames": {s: int(np.sum(labels == s)) for s in SETS}, "real": {}, "out": {}}
            jidx = np.arange(real_an["frames"])
            for s in SETS:
                row["real"][s] = dict(zip(("cpp_db", "hf_ratio_db"), descriptor_on(real16, jidx[labels == s])))
            for name, cond in CONDITIONS.items():
                path = outputs / "wav" / f"{key}_{cond}.wav"
                out_an = ob.analyse(load(path, ob.SR))
                index = ob.map_100fps(mapping, real_an["frames"], out_an["frames"])
                out16 = load(path, 16000)
                full = ob.compare(out_an, real_an, index)          # check against experiment 08
                ref = metrics[(singer, sid, cond)]
                check["d_raw"] = max(check["d_raw"], abs(full["d_raw"] - ref["d_raw"]))
                check["d_dyn"] = max(check["d_dyn"], abs(full["d_cmn"] - ref["d_cmn"]))
                desc = summarize(out16)
                check["cpp_db"] = max(check["cpp_db"], abs(desc["cpp_db"] - conds[(singer, sid, cond)]["cpp_db"]))
                check["hf_ratio_db"] = max(check["hf_ratio_db"],
                                           abs(desc["hf_ratio_db"] - conds[(singer, sid, cond)]["hf_ratio_db"]))
                row["out"][name] = {}
                for s in SETS:
                    keep = real_an["active"][: real_an["frames"]] & (labels == s)
                    if keep.sum() < MIN_FRAMES:
                        row["out"][name][s] = None
                        continue
                    a, b = out_an["cep"][index[keep]], real_an["cep"][: real_an["frames"]][keep]
                    cpp, hf = descriptor_on(out16, index[labels == s])
                    row["out"][name][s] = {"d_raw": ob.envelope_distance(a, b),
                                           "d_dyn": ob.envelope_distance(a, b, cmn=True),
                                           "frames_compared": int(keep.sum()), "cpp_db": cpp, "hf_ratio_db": hf}
            rows.append(row)
            print(f"{key}: breathy {row['frames']['breathy']} frames, other {row['frames']['other']}", flush=True)

    def collect(fn, s, singer=None):
        g = collections.defaultdict(list)
        for r in rows:
            if singer and r["singer"] != singer:
                continue
            try:
                v = fn(r, s)
            except (TypeError, KeyError):
                v = None
            if v is not None and np.isfinite(v):
                g[r["cluster"]].append(v)
        return g

    quantities = {
        "D a1-base": lambda r, s: r["out"]["a1"][s]["d_raw"] - r["out"]["base"][s]["d_raw"],
        "D a0.5-base": lambda r, s: r["out"]["a0.5"][s]["d_raw"] - r["out"]["base"][s]["d_raw"],
        "D_dyn a1-base": lambda r, s: r["out"]["a1"][s]["d_dyn"] - r["out"]["base"][s]["d_dyn"],
        "D_dyn a1-eqmatch": lambda r, s: r["out"]["a1"][s]["d_dyn"] - r["out"]["eqmatch"][s]["d_dyn"],
        "CPP a1-base": lambda r, s: r["out"]["a1"][s]["cpp_db"] - r["out"]["base"][s]["cpp_db"],
        "CPP base-real": lambda r, s: r["out"]["base"][s]["cpp_db"] - r["real"][s]["cpp_db"],
        "CPP a1-real": lambda r, s: r["out"]["a1"][s]["cpp_db"] - r["real"][s]["cpp_db"],
        "HF a1-base": lambda r, s: r["out"]["a1"][s]["hf_ratio_db"] - r["out"]["base"][s]["hf_ratio_db"],
        "HF base-real": lambda r, s: r["out"]["base"][s]["hf_ratio_db"] - r["real"][s]["hf_ratio_db"],
        "HF a1-real": lambda r, s: r["out"]["a1"][s]["hf_ratio_db"] - r["real"][s]["hf_ratio_db"],
    }
    singers = list(manifest["singers"])
    summary = {"pooled": {}, "per_singer": {}}
    for s in SETS:
        summary["pooled"][s] = {q: boot(collect(fn, s)) for q, fn in quantities.items() if collect(fn, s)}
        for singer in singers:
            summary["per_singer"].setdefault(singer, {})[s] = {
                q: [float(np.mean(v)), len(v)] for q, fn in quantities.items()
                for v in [[x for xs in collect(fn, s, singer).values() for x in xs]] if v}
    share = {}
    for singer in singers:
        fr = [r["frames"] for r in rows if r["singer"] == singer]
        b, o = sum(f["breathy"] for f in fr), sum(f["other"] for f in fr)
        share[singer] = {"breathy_share_of_sung_frames": b / (b + o), "phrases_without_breathy_frames":
                         sum(1 for f in fr if f["breathy"] == 0), "phrases": len(fr)}
    result = {"min_frames": MIN_FRAMES, "check_max_abs_difference_to_experiment_08": dict(check),
              "breathy_labels": share, "summary": summary, "rows": rows, "annotation_sha256": checksums}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "phonemes.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)

    L = ["Experiment 08 by GTSinger breathy label (breathy control, alpha = 1 unless stated)", "",
         "Check against experiment 08 (all active frames), largest absolute difference: "
         + ", ".join(f"{k} {v:.2e}" for k, v in check.items()), "",
         "Share of sung frames labelled breathy per singer:"]
    for singer, v in share.items():
        L.append(f"  {singer:11s} {v['breathy_share_of_sung_frames']:.2f} "
                 f"({v['phrases_without_breathy_frames']} of {v['phrases']} phrases without breathy frames)")
    for s in SETS:
        L += ["", f"== Sung phonemes {'labelled' if s == 'breathy' else 'not labelled'} breathy "
                  f"(pooled; mean [95% interval] n=phrases k=clusters) =="]
        for q, v in summary["pooled"][s].items():
            L.append(f"  {q:18s} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}] n={v[3]} k={v[4]}")
        L.append("  per singer (mean, n):")
        for singer in singers:
            ps = summary["per_singer"][singer][s]
            L.append(f"    {singer:11s} " + " | ".join(f"{q} {v[0]:+.2f} ({v[1]})" for q, v in ps.items()
                                                  if q in ("D a1-base", "D_dyn a1-base", "CPP a1-base", "HF a1-base",
                                                           "HF base-real")))
    text = "\n".join(L) + "\n"
    with open(out / "phonemes_report.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
