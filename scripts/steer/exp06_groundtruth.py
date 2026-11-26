"""Experiment 06A: how close does each method get to REAL technique recordings?

VocalSet has, for every singer and exercise, a plain (straight) take and takes of the same
exercise sung with each technique. For the five singers held out of adapter training:

  source     the singer's straight take of an exercise
  reference  the same singer's straight 'caro mio ben' excerpt (6 s), i.e. their normal voice
  truth      the same singer's real breathy / belt / vocal-fry / forte / piano take of the
             same exercise

Every output that keeps the source time line is compared with every real technique take
through one fixed DTW path per (exercise, technique), computed between the straight and
the technique recording (pitch, voicing and energy only, the same alignment used to build
the training pairs). The path does not depend on the output being scored.

Outputs on the source time line
  real_straight                 the straight recording itself (no conversion)
  real_straight_oracleeq_<k>    the straight recording, equalised to the real <k> take's
                                long-term spectrum (uses the answer; no neural network)
  base                          Seed-VC, adapter off
  ad_<k>_a1, ad_<k>_a0.5        adapter, label <k> (all five labels, so wrong labels are
                                scored against each real technique too)
  eqmatch_<k>                   base equalised to the long-term spectrum of ad_<k>_a1
  oracleeq_<k>                  base equalised to the real <k> take (uses the answer)
  cgm_<k>                       training-free multi-donor guidance, w = 4 (piano: forte
                                axis, w = -4); only for the scales_a exercise
Output on the real take's own time line
  carried_<k>                   the real <k> take used as the source (technique present in
                                the input): what the frozen converter does with it

Measures: envelope distance D (raw and CMN, see src/steer/objective.py), variance ratio,
CPP / HF ratio / level, F0 deviation from the source (RMVPE).

Outputs: results/exp06/groundtruth/ (wav/, align/, metrics.jsonl, metrics.json, summary.json)
Resumable: finished conversions are kept and skipped on a rerun.
"""
from __future__ import annotations

import json
import os
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

from probe_v0 import Session, log  # noqa: E402
from exp03_panel import SR, EQ_LIMIT_DB, ltas_db, third_octave_smooth, apply_eq, cut_clip  # noqa: E402
import prep_technique_pairs as pp  # noqa: E402

OUT = ROOT / "results/exp06/groundtruth"
VOCALSET = ROOT / "data/vocalset_full/FULL"
SRC_DIR = ROOT / "data/exp06_sources"
REF_DIR = ROOT / "data/exp06_refs"
ADAPTER = ROOT / "results/exp04/train/adapter_best.pt"
HELD_OUT = ("female1", "female3", "female9", "male3", "male9")
SETS = ("scales", "arpeggios")
LABELS = ("breathy", "belt", "vocal_fry", "forte", "piano")
FOLDER = {label: folder for folder, label in pp.TECHNIQUES.items()}
STRENGTHS = (1.0, 0.5)
REFERENCE_EXCERPT = "caro"
GUIDE_SIDS = ("scales_a",)
DONORS = ("female4", "female5", "female6", "female7")
GUIDE = {"breathy": ("breathy", "straight", 4.0), "belt": ("belt", "straight", 4.0),
         "vocal_fry": ("vocal_fry", "straight", 4.0), "forte": ("slow_forte", "slow_piano", 4.0),
         "piano": ("slow_forte", "slow_piano", -4.0)}
SANITY = ("data/vocalset/source_6s.wav", "data/style_pilot/identity_f3_6s.wav", "results/exp04/eval/wav/f3_row_base.wav")


def rel(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


def strength_tag(a):
    return f"a{a:g}"


def dropped_files():
    path = ROOT / "results/exp04/train/summary.json"
    if not path.exists():
        return set()
    return {f for f, _ in json.loads(path.read_text(encoding="utf-8")).get("dropped", [])}


def plan():
    dropped = dropped_files()
    items, excluded = [], []
    for singer in HELD_OUT:
        if not (VOCALSET / singer).is_dir():
            log(f"warning: {singer} not found under {VOCALSET}")
            continue
        for set_name in SETS:
            base = VOCALSET / singer / set_name
            for vowel, files in sorted(pp.takes(base / "straight").items()):
                sid = f"{set_name}_{vowel}"
                reals = {}
                for label in LABELS:
                    cands = [p for p in pp.takes(base / FOLDER[label]).get(vowel, []) if "(" not in p.name]
                    for p in cands:
                        if rel(p) in dropped:
                            excluded.append(rel(p))
                    cands = [p for p in cands if rel(p) not in dropped]
                    if cands:
                        reals[label] = cands
                items.append({"singer": singer, "sid": sid, "straight": files[0], "reals": reals})
    return items, excluded


def write_wave(path, wave, subtype="FLOAT"):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp" + path.suffix)
    sf.write(tmp, wave, SR, subtype=subtype)
    os.replace(tmp, path)


def eq_to(wave, target, freqs):
    gain = third_octave_smooth(freqs, ltas_db(target) - ltas_db(wave))
    gain[freqs < 60] = 0.0
    return apply_eq(wave, np.clip(gain, -EQ_LIMIT_DB, EQ_LIMIT_DB))


def stage_convert(session, extractor, adapter, items, wav_dir, align_dir):
    """All GPU work: references, sources, alignment, conversions. Skips finished files."""
    tap = session.tap
    refs = {}
    for singer in sorted({item["singer"] for item in items}):
        path = REF_DIR / f"{singer}_ref.wav"
        if not path.exists():
            files = sorted((VOCALSET / singer / "excerpts/straight").glob("*.wav"))
            chosen = [f for f in files if REFERENCE_EXCERPT in f.name] or files
            if not chosen:
                raise FileNotFoundError(f"no straight excerpt for {singer}")
            clip, _ = cut_clip(chosen[0], seconds=6.0)
            write_wave(path, clip, "PCM_16")
        refs[singer] = rel(path)

    donor_clips = {}
    for donor in DONORS:
        for tech in {t for pos, neg, _ in GUIDE.values() for t in (pos, neg)}:
            donor_clips[donor, tech] = f"data/exp03_refs/{donor}_{tech}.wav"

    def convert(source, reference, dest, amounts=None):
        if dest.exists():
            return False
        if amounts:
            adapter.set(**amounts)
        try:
            session.convert(source, reference, dest)
        finally:
            adapter.off()
        return True

    total = len(items)
    for n, item in enumerate(items, 1):
        singer, sid = item["singer"], item["sid"]
        key = f"{singer}_{sid}"
        started = time.perf_counter()
        src_path = SRC_DIR / f"{key}_straight.wav"
        if not src_path.exists():
            write_wave(src_path, pp.load_trimmed(item["straight"]))
        align_path = align_dir / f"{key}.json"
        if align_path.exists():
            chosen = json.loads(align_path.read_text(encoding="utf-8"))
        else:
            import librosa
            src_wave, _ = librosa.load(str(src_path), sr=SR, mono=True)
            src = extractor.analyse(src_wave)
            chosen = {}
            for label, cands in item["reals"].items():
                best = None
                for cand in cands:
                    wave = pp.load_trimmed(cand)
                    tgt = extractor.analyse(wave)
                    mapping, diag = pp.align(src, tgt, pitch_weight=0.3 if label == "vocal_fry" else 1.0)
                    score = (abs(diag["key_offset"]), diag["path_cost"])
                    if best is None or score < best[0]:
                        best = (score, cand, wave, mapping, diag)
                _, cand, wave, mapping, diag = best
                real_path = SRC_DIR / f"{key}_{label}.wav"
                write_wave(real_path, wave)
                chosen[label] = {"file": rel(cand), "trimmed": rel(real_path), "mapping50": mapping.tolist(), **diag}
            align_path.parent.mkdir(parents=True, exist_ok=True)
            align_path.write_text(json.dumps(chosen), encoding="utf-8")
        source, reference = rel(src_path), refs[singer]
        made = 0
        made += convert(source, reference, wav_dir / f"{key}_base.wav")
        for label in LABELS:
            for a in STRENGTHS:
                made += convert(source, reference, wav_dir / f"{key}_ad_{label}_{strength_tag(a)}.wav", {label: a})
        for label, info in chosen.items():
            made += convert(info["trimmed"], reference, wav_dir / f"{key}_carried_{label}.wav")
        if sid in GUIDE_SIDS:
            todo = [label for label in LABELS if not (wav_dir / f"{key}_cgm_{label}.wav").exists()]
            if todo:
                packs, scratch = {}, wav_dir / "_packs"
                scratch.mkdir(parents=True, exist_ok=True)
                needed = {t for label in todo for t in GUIDE[label][:2]}
                for donor in DONORS:
                    for tech in sorted(needed):
                        tap.start_capture()
                        session.convert(source, donor_clips[donor, tech], scratch / f"{key}_{donor}_{tech}.wav")
                        packs[donor, tech] = tap.stop_capture()
                for label in todo:
                    pos, neg, weight = GUIDE[label]
                    tap.set_guidance([packs[d, pos] for d in DONORS], [packs[d, neg] for d in DONORS], weight)
                    try:
                        session.convert(source, reference, wav_dir / f"{key}_cgm_{label}.wav")
                        made += 1
                    except Exception as error:  # e.g. out of GPU memory on a long take
                        traceback.print_exc()
                        (wav_dir / f"{key}_cgm_{label}.error.txt").write_text(f"{type(error).__name__}: {error}",
                                                                               encoding="utf-8")
                        if hasattr(session.torch, "cuda") and session.torch.cuda.is_available():
                            session.torch.cuda.empty_cache()
                    finally:
                        tap.set_guidance(None)
                del packs
        log(f"[{n}/{total}] {key}: {made} new conversions, {time.perf_counter() - started:.0f}s")


def stage_equalise(items, wav_dir, align_dir):
    import librosa
    import numpy as np
    freqs = np.fft.rfftfreq(2048, 1.0 / SR)
    for item in items:
        key = f"{item['singer']}_{item['sid']}"
        chosen = json.loads((align_dir / f"{key}.json").read_text(encoding="utf-8"))
        base, _ = librosa.load(str(wav_dir / f"{key}_base.wav"), sr=SR, mono=True)
        straight, _ = librosa.load(str(SRC_DIR / f"{key}_straight.wav"), sr=SR, mono=True)
        for label in LABELS:
            path = wav_dir / f"{key}_eqmatch_{label}.wav"
            if not path.exists():
                neural, _ = librosa.load(str(wav_dir / f"{key}_ad_{label}_a1.wav"), sr=SR, mono=True)
                write_wave(path, eq_to(base, neural, freqs), "PCM_16")
        for label, info in chosen.items():
            real, _ = librosa.load(str(ROOT / info["trimmed"]), sr=SR, mono=True)
            path = wav_dir / f"{key}_oracleeq_{label}.wav"
            if not path.exists():
                write_wave(path, eq_to(base, real, freqs), "PCM_16")
            path = wav_dir / f"{key}_real_straight_oracleeq_{label}.wav"
            if not path.exists():
                write_wave(path, eq_to(straight, real, freqs), "PCM_16")


def stage_measure(session, items, wav_dir, align_dir):
    import librosa
    from steer import objective as ob
    from steer.metrics import summarize, f0_agreement

    rows, conditions = [], []
    stream = (OUT / "metrics.jsonl").open("w", encoding="utf-8")

    def descriptors(path):
        w16 = session.load16k(path)
        return w16, summarize(w16)

    for item in items:
        singer, sid = item["singer"], item["sid"]
        key = f"{singer}_{sid}"
        chosen = json.loads((align_dir / f"{key}.json").read_text(encoding="utf-8"))
        reals = {}
        for label, info in chosen.items():
            wave, _ = librosa.load(str(ROOT / info["trimmed"]), sr=SR, mono=True)
            _, desc = descriptors(ROOT / info["trimmed"])
            reals[label] = {"an": ob.analyse(wave), "desc": desc, "info": info}
        src_path = SRC_DIR / f"{key}_straight.wav"
        src_frames = ob.analyse(librosa.load(str(src_path), sr=SR, mono=True)[0])["frames"]
        src16 = session.load16k(src_path)
        src_f0 = session.f0(src16)
        timeline = [("real_straight", src_path, {"method": "real", "label": None})]
        timeline += [("base", wav_dir / f"{key}_base.wav", {"method": "base", "label": None})]
        for label in LABELS:
            for a in STRENGTHS:
                timeline.append((f"ad_{label}_{strength_tag(a)}", wav_dir / f"{key}_ad_{label}_{strength_tag(a)}.wav",
                                 {"method": "adapter", "label": label, "strength": a}))
            timeline.append((f"eqmatch_{label}", wav_dir / f"{key}_eqmatch_{label}.wav",
                             {"method": "eqmatch", "label": label}))
            timeline.append((f"cgm_{label}", wav_dir / f"{key}_cgm_{label}.wav", {"method": "guidance", "label": label}))
        for label in chosen:
            timeline.append((f"oracleeq_{label}", wav_dir / f"{key}_oracleeq_{label}.wav",
                             {"method": "oracleeq", "label": label}))
            timeline.append((f"real_straight_oracleeq_{label}", wav_dir / f"{key}_real_straight_oracleeq_{label}.wav",
                             {"method": "real_oracleeq", "label": label}))
        outputs = [(c, p, i, "source") for c, p, i in timeline if p.exists()]
        outputs += [(f"carried_{label}", wav_dir / f"{key}_carried_{label}.wav", {"method": "carried", "label": label},
                     "own") for label in chosen if (wav_dir / f"{key}_carried_{label}.wav").exists()]
        for cond, path, info, line in outputs:
            try:
                wave, _ = librosa.load(str(path), sr=SR, mono=True)
                an = ob.analyse(wave)
                w16, desc = descriptors(path)
                extra = {}
                if line == "source" and info["method"] in ("base", "adapter", "guidance"):
                    extra = f0_agreement(session.f0(w16), src_f0)
                own = reals[info["label"]]["an"]["frames"] if line == "own" else src_frames
                conditions.append({"singer": singer, "sid": sid, "condition": cond, "file": rel(path),
                                   "timeline": line, "length_ratio": an["frames"] / max(1, own),
                                   **info, **desc, **extra})
                for label, real in reals.items():
                    if line == "own" and label != info["label"]:
                        continue
                    index = None
                    if line == "source":
                        index = ob.map_100fps(real["info"]["mapping50"], real["an"]["frames"], an["frames"])
                    d = ob.compare(an, real["an"], index)
                    row = {"singer": singer, "sid": sid, "condition": cond, "truth": label, **info, **d,
                           "real_file": real["info"]["file"], "key_offset": real["info"]["key_offset"],
                           "path_cost": real["info"]["path_cost"]}
                    for name in ("cpp_db", "hf_ratio_db", "rms_dbfs"):
                        if desc.get(name) is not None and real["desc"].get(name) is not None:
                            row[f"{name}_minus_truth"] = desc[name] - real["desc"][name]
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
            except Exception as error:
                traceback.print_exc()
                rows.append({"singer": singer, "sid": sid, "condition": cond, "error": f"{type(error).__name__}: {error}"})
        stream.flush()
    stream.close()
    return rows, conditions


def summarise(rows):
    import numpy as np
    good = [r for r in rows if "d_raw" in r and r["d_raw"] is not None]
    by = {}
    for r in good:
        by.setdefault((r["singer"], r["sid"], r["truth"]), {})[r["condition"]] = r

    def table(metric):
        out = {}
        for truth in LABELS:
            cells = {}
            for (singer, sid, t), conds in by.items():
                if t != truth or "base" not in conds:
                    continue
                pick = {"real_straight": "real_straight", "real_straight_oracleeq": f"real_straight_oracleeq_{truth}",
                        "base": "base", "static_bound": "base", "adapter_a1": f"ad_{truth}_a1",
                        "adapter_a0.5": f"ad_{truth}_a0.5", "eqmatch": f"eqmatch_{truth}",
                        "oracleeq": f"oracleeq_{truth}", "guidance": f"cgm_{truth}", "carried": f"carried_{truth}"}
                for name, cond in pick.items():
                    if cond in conds:
                        value = conds[cond]["d_cmn" if name == "static_bound" else metric]
                        cells.setdefault(name, []).append((singer, value))
                wrong = [conds[f"ad_{j}_a1"][metric] for j in LABELS if j != truth and f"ad_{j}_a1" in conds]
                if wrong:
                    cells.setdefault("adapter_wrong_labels", []).append((singer, float(np.mean(wrong))))
            out[truth] = {name: {"mean": float(np.mean([v for _, v in vals])), "n": len(vals),
                                 "per_singer": {s: float(np.mean([v for ss, v in vals if ss == s]))
                                                for s in sorted({ss for ss, _ in vals})}}
                          for name, vals in cells.items()}
        return out

    confusion = {}
    for metric in ("d_raw", "d_cmn"):
        matrix, closest = {}, {}
        for truth in LABELS:
            items = [conds for (s, sid, t), conds in by.items() if t == truth and all(f"ad_{j}_a1" in conds for j in LABELS)]
            matrix[truth] = {j: float(np.mean([c[f"ad_{j}_a1"][metric] for c in items])) if items else None for j in LABELS}
            wins = [min(LABELS, key=lambda j: c[f"ad_{j}_a1"][metric]) for c in items]
            closest[truth] = {j: wins.count(j) for j in LABELS} | {"n": len(items)}
        confusion[metric] = {"mean_distance": matrix, "closest_label_counts": closest}
    return {"raw": table("d_raw"), "cmn": table("d_cmn"), "label_confusion": confusion,
            "notes": "static_bound = D_cmn(base, truth), the smallest raw distance any fixed per-band gain on the "
                     "base output could reach; compare it with the raw distance of the other conditions."}


def main():
    import numpy as np
    import soundfile as sf
    import torch
    from steer.adapter import TechniqueAdapter

    if (OUT / "summary.json").exists():
        log(f"{OUT} already has summary.json; nothing to do")
        return 0
    if (OUT / "failure.json").exists():
        os.replace(OUT / "failure.json", OUT / f"failure_{int(time.time())}.json")
    if not ADAPTER.exists():
        raise FileNotFoundError(f"{ADAPTER} not found")
    wav_dir, align_dir = OUT / "wav", OUT / "align"
    wav_dir.mkdir(parents=True, exist_ok=True)
    items, excluded = plan()
    log(f"{len(items)} exercises from {len(HELD_OUT)} held-out singers; "
        f"{sum(len(i['reals']) for i in items)} exercise-technique pairs; {len(excluded)} takes excluded "
        f"(worst-aligned, as in training)")

    session = Session()
    extractor = pp.Extractor(session)
    adapter = TechniqueAdapter.from_file(session.model.cfm.estimator, ADAPTER)
    adapter.off()
    log(f"adapter loaded: {adapter.techniques}")
    if list(adapter.techniques) != list(LABELS):
        raise ValueError(f"adapter techniques {adapter.techniques} differ from {LABELS}")

    sanity = {}
    source, target, previous = SANITY
    if (ROOT / previous).exists():
        check = wav_dir / "_sanity_f3_row_base.wav"
        if not check.exists():
            session.convert(source, target, check)
        a, _ = sf.read(str(check), dtype="int16")
        b, _ = sf.read(str(ROOT / previous), dtype="int16")
        sanity["base_matches_exp04"] = bool(a.shape == b.shape and np.array_equal(a, b))
        log(f"environment check, exp04 base reproduced bit for bit: {sanity['base_matches_exp04']}")

    started = time.perf_counter()
    stage_convert(session, extractor, adapter, items, wav_dir, align_dir)
    convert_minutes = (time.perf_counter() - started) / 60
    log(f"conversions finished in {convert_minutes:.1f} min; equalising")
    stage_equalise(items, wav_dir, align_dir)
    log("measuring")
    rows, conditions = stage_measure(session, items, wav_dir, align_dir)
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    (OUT / "conditions.json").write_text(json.dumps(conditions, indent=0), encoding="utf-8")
    summary = {"status": "completed", "held_out": HELD_OUT, "exercises": len(items),
               "excluded_takes": excluded, "rows": len(rows), "errors": sum("error" in r for r in rows),
               "guidance_errors": sorted(p.name for p in wav_dir.glob("*.error.txt")),
               "convert_minutes": convert_minutes, "sanity": sanity, "seed": session.seed,
               "settings": session.settings, **summarise(rows)}
    if torch.cuda.is_available():
        summary["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for truth in LABELS:
        t = summary["raw"].get(truth, {})
        fmt = lambda k: f"{t[k]['mean']:.2f}" if k in t else "-"  # noqa: E731
        log(f"{truth:9s} D_raw  straight {fmt('real_straight')}  base {fmt('base')}  adapter {fmt('adapter_a1')}  "
            f"eqmatch {fmt('eqmatch')}  oracleEQ {fmt('oracleeq')}  static bound {fmt('static_bound')}  "
            f"carried {fmt('carried')}")
    return 0


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        code = main()
    except Exception as error:
        traceback.print_exc()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "failure.json").write_text(json.dumps({"error_type": type(error).__name__, "error": str(error)}),
                                          encoding="utf-8")
        code = 1
    log(f"total wall {time.perf_counter() - started:.1f}s, exit {code}")
    raise SystemExit(code)
