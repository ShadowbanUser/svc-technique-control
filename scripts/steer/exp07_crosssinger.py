"""Experiment 07: conversion between different held-out singers, scored against real recordings,
plus the costs of each control on the experiment-06 outputs.

Part A, cross-singer ground truth. Among the five held-out singers, singers of the same sex
form ordered pairs A -> B (female1/female3/female9: six pairs; male3/male9: two pairs). For
every scale and arpeggio exercise, singer A's plain take is converted into singer B's voice
(reference: B's plain 'Caro mio ben' excerpt, the one used in experiment 06) and compared with
B's real technique take of the same exercise through one fixed DTW path between A's plain take
and B's technique take. Outputs: base, adapter at alpha = 1 with all five labels, matched EQ
of each adapter output, oracle EQ of base to B's real take. Same measures as experiment 06.

Part B, costs on the experiment-06 outputs (self-conversion):
  - speaker similarity: cosine of CAM++ embeddings between each output and its reference;
  - pitch tracked the same way for the real vocal-fry takes and for the vocal-fry outputs:
    mean absolute deviation (cents) from the plain take, along the experiment-06 DTW path for
    the real takes (key offset removed) and frame by frame for the outputs.

Reuses experiment-06 sources, references and alignments. Outputs: results/exp07/.
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

from probe_v0 import Session, log, cosine  # noqa: E402
import prep_technique_pairs as pp  # noqa: E402
import exp06_groundtruth as g6  # noqa: E402

OUT = ROOT / "results/exp07"
GT6 = ROOT / "results/exp06/groundtruth"
LABELS = g6.LABELS
GROUPS = (("female1", "female3", "female9"), ("male3", "male9"))
PAIRS = [(a, b) for group in GROUPS for a in group for b in group if a != b]


def rel(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


def plan():
    items = []
    for a, b in PAIRS:
        for path in sorted((GT6 / "align").glob(f"{b}_*.json")):
            sid = path.stem[len(b) + 1:]
            src = g6.SRC_DIR / f"{a}_{sid}_straight.wav"
            if not src.exists():
                continue
            chosen = json.loads(path.read_text(encoding="utf-8"))
            if chosen:
                items.append({"a": a, "b": b, "sid": sid, "source": src, "truth": chosen})
    return items


def stage_convert(session, extractor, adapter, items, wav_dir, align_dir):
    import librosa
    total = len(items)
    for n, it in enumerate(items, 1):
        key = f"{it['a']}_to_{it['b']}_{it['sid']}"
        started = time.perf_counter()
        align_path = align_dir / f"{key}.json"
        if not align_path.exists():
            src = extractor.analyse(librosa.load(str(it["source"]), sr=g6.SR, mono=True)[0])
            out = {}
            for label, info in it["truth"].items():
                tgt = extractor.analyse(librosa.load(str(ROOT / info["trimmed"]), sr=g6.SR, mono=True)[0])
                mapping, diag = pp.align(src, tgt, pitch_weight=0.3 if label == "vocal_fry" else 1.0)
                out[label] = {"file": info["file"], "trimmed": info["trimmed"], "mapping50": mapping.tolist(), **diag}
            align_dir.mkdir(parents=True, exist_ok=True)
            align_path.write_text(json.dumps(out), encoding="utf-8")
        source, reference = rel(it["source"]), rel(g6.REF_DIR / f"{it['b']}_ref.wav")
        made = 0
        dest = wav_dir / f"{key}_base.wav"
        if not dest.exists():
            session.convert(source, reference, dest)
            made += 1
        for label in LABELS:
            dest = wav_dir / f"{key}_ad_{label}_a1.wav"
            if dest.exists():
                continue
            adapter.set(**{label: 1.0})
            try:
                session.convert(source, reference, dest)
            finally:
                adapter.off()
            made += 1
        log(f"[{n}/{total}] {key}: {made} new conversions, {time.perf_counter() - started:.0f}s")


def stage_equalise(items, wav_dir, align_dir):
    import librosa
    freqs = np.fft.rfftfreq(2048, 1.0 / g6.SR)
    for it in items:
        key = f"{it['a']}_to_{it['b']}_{it['sid']}"
        chosen = json.loads((align_dir / f"{key}.json").read_text(encoding="utf-8"))
        base, _ = librosa.load(str(wav_dir / f"{key}_base.wav"), sr=g6.SR, mono=True)
        for label in LABELS:
            path = wav_dir / f"{key}_eqmatch_{label}.wav"
            if not path.exists():
                neural, _ = librosa.load(str(wav_dir / f"{key}_ad_{label}_a1.wav"), sr=g6.SR, mono=True)
                g6.write_wave(path, g6.eq_to(base, neural, freqs), "PCM_16")
        for label, info in chosen.items():
            path = wav_dir / f"{key}_oracleeq_{label}.wav"
            if not path.exists():
                real, _ = librosa.load(str(ROOT / info["trimmed"]), sr=g6.SR, mono=True)
                g6.write_wave(path, g6.eq_to(base, real, freqs), "PCM_16")


def stage_measure(session, items, wav_dir, align_dir):
    import librosa
    from steer import objective as ob
    from steer.metrics import summarize, f0_agreement
    rows, conditions = [], []
    embed_cache = {}

    def embedding(path):
        if path not in embed_cache:
            embed_cache[path] = session.embed(session.load16k(path))
        return embed_cache[path]

    for it in items:
        key = f"{it['a']}_to_{it['b']}_{it['sid']}"
        chosen = json.loads((align_dir / f"{key}.json").read_text(encoding="utf-8"))
        reals = {}
        for label, info in chosen.items():
            wave, _ = librosa.load(str(ROOT / info["trimmed"]), sr=g6.SR, mono=True)
            reals[label] = {"an": ob.analyse(wave), "desc": summarize(session.load16k(ROOT / info["trimmed"])), "info": info}
        src16 = session.load16k(it["source"])
        src_f0 = session.f0(src16)
        ref_path = g6.REF_DIR / f"{it['b']}_ref.wav"
        outputs = [("base", wav_dir / f"{key}_base.wav", {"method": "base", "label": None})]
        for label in LABELS:
            outputs.append((f"ad_{label}_a1", wav_dir / f"{key}_ad_{label}_a1.wav",
                            {"method": "adapter", "label": label, "strength": 1.0}))
            outputs.append((f"eqmatch_{label}", wav_dir / f"{key}_eqmatch_{label}.wav", {"method": "eqmatch", "label": label}))
        for label in chosen:
            outputs.append((f"oracleeq_{label}", wav_dir / f"{key}_oracleeq_{label}.wav", {"method": "oracleeq", "label": label}))
        for cond, path, info in outputs:
            if not path.exists():
                continue
            try:
                wave, _ = librosa.load(str(path), sr=g6.SR, mono=True)
                an = ob.analyse(wave)
                w16 = session.load16k(path)
                desc = summarize(w16)
                extra = {}
                if info["method"] in ("base", "adapter"):
                    extra = f0_agreement(session.f0(w16), src_f0)
                    extra["similarity"] = cosine(embedding(path), embedding(ref_path))
                conditions.append({"singer": it["b"], "source_singer": it["a"], "sid": f"{it['a']}:{it['sid']}",
                                   "condition": cond, "file": rel(path), **info, **desc, **extra})
                for label, real in reals.items():
                    index = ob.map_100fps(real["info"]["mapping50"], real["an"]["frames"], an["frames"])
                    d = ob.compare(an, real["an"], index)
                    row = {"singer": it["b"], "source_singer": it["a"], "sid": f"{it['a']}:{it['sid']}", "condition": cond,
                           "truth": label, **info, **d, "real_file": real["info"]["file"],
                           "key_offset": real["info"]["key_offset"], "path_cost": real["info"]["path_cost"]}
                    for name in ("cpp_db", "hf_ratio_db", "rms_dbfs"):
                        if desc.get(name) is not None and real["desc"].get(name) is not None:
                            row[f"{name}_minus_truth"] = desc[name] - real["desc"][name]
                    rows.append(row)
            except Exception as error:
                traceback.print_exc()
                rows.append({"singer": it["b"], "sid": f"{it['a']}:{it['sid']}", "condition": cond,
                             "error": f"{type(error).__name__}: {error}"})
    return rows, conditions


def stage_costs(session, extractor):
    """Speaker similarity of experiment-06 outputs, and pitch tracked the same way for real and generated fry."""
    import librosa
    conds = json.loads((GT6 / "conditions.json").read_text(encoding="utf-8"))
    sims = []
    ref_embed = {}
    for c in conds:
        if c["method"] not in ("base", "adapter", "eqmatch", "carried"):
            continue
        ref = g6.REF_DIR / f"{c['singer']}_ref.wav"
        if c["singer"] not in ref_embed:
            ref_embed[c["singer"]] = session.embed(session.load16k(ref))
        e = session.embed(session.load16k(ROOT / c["file"]))
        sims.append({k: c.get(k) for k in ("singer", "sid", "condition", "method", "label", "strength")} |
                    {"similarity": cosine(e, ref_embed[c["singer"]])})
    pitch = []
    for path in sorted((GT6 / "align").glob("*.json")):
        chosen = json.loads(path.read_text(encoding="utf-8"))
        if "vocal_fry" not in chosen:
            continue
        key = path.stem
        src = extractor.analyse(librosa.load(str(g6.SRC_DIR / f"{key}_straight.wav"), sr=g6.SR, mono=True)[0])
        info = chosen["vocal_fry"]
        real = extractor.analyse(librosa.load(str(ROOT / info["trimmed"]), sr=g6.SR, mono=True)[0])
        mapping = np.asarray(info["mapping50"], dtype=np.int64)
        n = min(len(mapping), len(real["st50"]))
        st_r, v_r = np.asarray(real["st50"])[:n], np.asarray(real["v50"])[:n].astype(bool)
        idx = np.clip(mapping[:n], 0, len(src["st50"]) - 1)
        st_s, v_s = np.asarray(src["st50"])[idx], np.asarray(src["v50"])[idx].astype(bool)
        both = v_r & v_s
        rec = {"item": key, "real_voiced_share": float(v_r.mean()) if n else None,
               "real_dev_cents": float(np.mean(np.abs(st_r[both] - st_s[both] - info["key_offset"])) * 100) if both.any() else None}
        for tag in ("a1", "a0.5"):
            out_path = g6.OUT / "wav" / f"{key}_ad_vocal_fry_{tag}.wav"
            if not out_path.exists():
                continue
            out = extractor.analyse(librosa.load(str(out_path), sr=g6.SR, mono=True)[0])
            m = min(len(out["st50"]), len(src["st50"]))
            vo, vs = np.asarray(out["v50"])[:m].astype(bool), np.asarray(src["v50"])[:m].astype(bool)
            b = vo & vs
            rec[f"out_{tag}_dev_cents"] = float(np.mean(np.abs(np.asarray(out["st50"])[:m][b] - np.asarray(src["st50"])[:m][b])) * 100) if b.any() else None
            rec[f"out_{tag}_voiced_share"] = float(vo.mean())
        pitch.append(rec)
    return sims, pitch


def main():
    from steer.adapter import TechniqueAdapter
    import torch
    if (OUT / "summary.json").exists():
        log(f"{OUT} already has summary.json; nothing to do")
        return 0
    if (OUT / "failure.json").exists():
        os.replace(OUT / "failure.json", OUT / f"failure_{int(time.time())}.json")
    if not (GT6 / "summary.json").exists():
        raise FileNotFoundError("experiment 06 results are needed first")
    wav_dir, align_dir = OUT / "crosssinger/wav", OUT / "crosssinger/align"
    wav_dir.mkdir(parents=True, exist_ok=True)
    items = plan()
    log(f"{len(items)} cross-singer items from {len(PAIRS)} ordered singer pairs; "
        f"{sum(len(i['truth']) for i in items)} item-technique pairs")
    session = Session()
    extractor = pp.Extractor(session)
    adapter = TechniqueAdapter.from_file(session.model.cfm.estimator, g6.ADAPTER)
    adapter.off()
    started = time.perf_counter()
    stage_convert(session, extractor, adapter, items, wav_dir, align_dir)
    minutes = (time.perf_counter() - started) / 60
    log(f"conversions finished in {minutes:.1f} min; equalising")
    stage_equalise(items, wav_dir, align_dir)
    log("measuring cross-singer outputs")
    rows, conditions = stage_measure(session, items, wav_dir, align_dir)
    (OUT / "crosssinger/metrics.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    (OUT / "crosssinger/conditions.json").write_text(json.dumps(conditions, indent=0), encoding="utf-8")
    log("measuring costs on experiment-06 outputs")
    sims, pitch = stage_costs(session, extractor)
    (OUT / "costs_similarity.json").write_text(json.dumps(sims, indent=0), encoding="utf-8")
    (OUT / "costs_fry_pitch.json").write_text(json.dumps(pitch, indent=0), encoding="utf-8")
    summary = {"status": "completed", "pairs": PAIRS, "items": len(items), "rows": len(rows),
               "errors": sum("error" in r for r in rows), "convert_minutes": minutes,
               "crosssinger": g6.summarise(rows)}
    if torch.cuda.is_available():
        summary["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for truth in LABELS:
        t = summary["crosssinger"]["raw"].get(truth, {})
        fmt = lambda k: f"{t[k]['mean']:.2f}" if k in t else "-"  # noqa: E731
        log(f"{truth:9s} cross-singer D_raw base {fmt('base')}  adapter {fmt('adapter_a1')}  eqmatch {fmt('eqmatch')}  "
            f"oracleEQ {fmt('oracleeq')}  static bound {fmt('static_bound')}")
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
