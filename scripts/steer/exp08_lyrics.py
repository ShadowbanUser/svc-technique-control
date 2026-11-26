"""Experiment 08: the breathy control on sung lyric phrases (GTSinger, English), scored against real recordings.

GTSinger records every phrase twice by the same singer: once plainly (Control_Group) and once with
breathy technique (Breathy_Group), same lyrics and melody. Three English singers, 30 phrase pairs each,
fixed in data/exp08_manifest.json (chosen from file sizes only, before any audio was heard or measured).
None of these singers or songs was used for training.

Self-conversion, as in experiment 06: the plain phrase is converted into the same singer's voice
(reference: two plain phrases of a song not used as a test item) and compared with the real breathy
phrase through one fixed DTW path between the plain and the breathy phrase.

Conditions per phrase: base; adapter at alpha = 1 with all five labels; adapter breathy at alpha = 0.5;
matched EQ of the breathy output; oracle EQ of base to the real breathy phrase; the real plain phrase
and its oracle EQ (how far plain singing is from breathy singing); 'carried': the real breathy phrase
used as the source (does the converter keep breathiness that is already in the input?).

Data: data/gtsinger/<path in the GTSinger repository>. Outputs: results/exp08/. Resumable.
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

OUT = ROOT / "results/exp08"
DATA = ROOT / "data/gtsinger"
REF_DIR = ROOT / "data/exp08_refs"
MANIFEST = ROOT / "data/exp08_manifest.json"
LABELS = g6.LABELS
TRUTH = "breathy"
REF_SECONDS = 15.0


def rel(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


def plan():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items, missing = [], []
    for singer, entry in manifest["singers"].items():
        for n, pair in enumerate(entry["pairs"]):
            control, breathy = DATA / pair["control"], DATA / pair["breathy"]
            if control.exists() and breathy.exists():
                items.append({"singer": singer, "sid": f"{n:02d}", "song": pair["song"], "index": pair["index"],
                              "control": control, "breathy": breathy})
            else:
                missing.append(pair["control"])
    return manifest, items, missing


def make_refs(manifest):
    import librosa
    REF_DIR.mkdir(parents=True, exist_ok=True)
    for singer, entry in manifest["singers"].items():
        path = REF_DIR / f"{singer}_ref.wav"
        if path.exists():
            continue
        parts = [librosa.load(str(DATA / p), sr=g6.SR, mono=True)[0] for p in entry["reference"]]
        wave = np.concatenate(parts)[: int(REF_SECONDS * g6.SR)]
        g6.write_wave(path, wave, "PCM_16")


def key_of(it):
    return f"{it['singer']}_{it['sid']}"


def stage_convert(session, extractor, adapter, items, wav_dir, align_dir):
    import librosa
    total = len(items)
    for n, it in enumerate(items, 1):
        key, started, made = key_of(it), time.perf_counter(), 0
        align_path = align_dir / f"{key}.json"
        if not align_path.exists():
            src = extractor.analyse(librosa.load(str(it["control"]), sr=g6.SR, mono=True)[0])
            tgt = extractor.analyse(librosa.load(str(it["breathy"]), sr=g6.SR, mono=True)[0])
            mapping, diag = pp.align(src, tgt, pitch_weight=1.0)
            align_dir.mkdir(parents=True, exist_ok=True)
            align_path.write_text(json.dumps({TRUTH: {"file": rel(it["breathy"]), "trimmed": rel(it["breathy"]),
                                                      "mapping50": mapping.tolist(), **diag}}), encoding="utf-8")
        source, reference = rel(it["control"]), rel(REF_DIR / f"{it['singer']}_ref.wav")
        jobs = [(f"{key}_base.wav", source, None, 0.0), (f"{key}_carried_{TRUTH}.wav", rel(it["breathy"]), None, 0.0)]
        jobs += [(f"{key}_ad_{label}_a1.wav", source, label, 1.0) for label in LABELS]
        jobs += [(f"{key}_ad_{TRUTH}_a0.5.wav", source, TRUTH, 0.5)]
        for name, src_path, label, strength in jobs:
            dest = wav_dir / name
            if dest.exists():
                continue
            if label is not None:
                adapter.set(**{label: strength})
            try:
                session.convert(src_path, reference, dest)
            finally:
                adapter.off()
            made += 1
        log(f"[{n}/{total}] {key} ({it['song']} {it['index']}): {made} new conversions, "
            f"{time.perf_counter() - started:.0f}s")


def stage_equalise(items, wav_dir):
    import librosa
    freqs = np.fft.rfftfreq(2048, 1.0 / g6.SR)
    for it in items:
        key = key_of(it)
        base = librosa.load(str(wav_dir / f"{key}_base.wav"), sr=g6.SR, mono=True)[0]
        real = librosa.load(str(it["breathy"]), sr=g6.SR, mono=True)[0]
        plain = librosa.load(str(it["control"]), sr=g6.SR, mono=True)[0]
        jobs = [(f"{key}_eqmatch_{TRUTH}.wav", base, f"{key}_ad_{TRUTH}_a1.wav"),
                (f"{key}_oracleeq_{TRUTH}.wav", base, real),
                (f"{key}_real_straight_oracleeq_{TRUTH}.wav", plain, real)]
        for name, wave, target in jobs:
            path = wav_dir / name
            if path.exists():
                continue
            if isinstance(target, str):
                target = librosa.load(str(wav_dir / target), sr=g6.SR, mono=True)[0]
            g6.write_wave(path, g6.eq_to(wave, target, freqs), "PCM_16")


def stage_measure(session, items, wav_dir, align_dir):
    import librosa
    from steer import objective as ob
    from steer.metrics import f0_agreement, summarize
    rows, conditions, ref_embed = [], [], {}

    def descriptors(path):
        w16 = session.load16k(path)
        return w16, summarize(w16)

    for it in items:
        key = key_of(it)
        info = json.loads((align_dir / f"{key}.json").read_text(encoding="utf-8"))[TRUTH]
        real_an = ob.analyse(librosa.load(str(it["breathy"]), sr=g6.SR, mono=True)[0])
        _, real_desc = descriptors(it["breathy"])
        src16 = session.load16k(it["control"])
        src_f0 = session.f0(src16)
        if it["singer"] not in ref_embed:
            ref_embed[it["singer"]] = session.embed(session.load16k(REF_DIR / f"{it['singer']}_ref.wav"))
        timeline = [("real_straight", it["control"], {"method": "real", "label": None}),
                    ("base", wav_dir / f"{key}_base.wav", {"method": "base", "label": None})]
        timeline += [(f"ad_{label}_a1", wav_dir / f"{key}_ad_{label}_a1.wav",
                      {"method": "adapter", "label": label, "strength": 1.0}) for label in LABELS]
        timeline += [(f"ad_{TRUTH}_a0.5", wav_dir / f"{key}_ad_{TRUTH}_a0.5.wav",
                      {"method": "adapter", "label": TRUTH, "strength": 0.5}),
                     (f"eqmatch_{TRUTH}", wav_dir / f"{key}_eqmatch_{TRUTH}.wav", {"method": "eqmatch", "label": TRUTH}),
                     (f"oracleeq_{TRUTH}", wav_dir / f"{key}_oracleeq_{TRUTH}.wav", {"method": "oracleeq", "label": TRUTH}),
                     (f"real_straight_oracleeq_{TRUTH}", wav_dir / f"{key}_real_straight_oracleeq_{TRUTH}.wav",
                      {"method": "real_oracleeq", "label": TRUTH})]
        outputs = [(c, p, i, "source") for c, p, i in timeline]
        outputs.append((f"carried_{TRUTH}", wav_dir / f"{key}_carried_{TRUTH}.wav", {"method": "carried", "label": TRUTH}, "own"))
        for cond, path, meta, line in outputs:
            if not Path(path).exists():
                continue
            try:
                an = ob.analyse(librosa.load(str(path), sr=g6.SR, mono=True)[0])
                w16, desc = descriptors(path)
                extra = {}
                if meta["method"] in ("base", "adapter", "carried"):
                    if line == "source":
                        extra = f0_agreement(session.f0(w16), src_f0)
                    extra["similarity"] = cosine(session.embed(w16), ref_embed[it["singer"]])
                conditions.append({"singer": it["singer"], "sid": it["sid"], "song": it["song"], "condition": cond,
                                   "file": rel(path), "timeline": line, **meta, **desc, **extra})
                index = ob.map_100fps(info["mapping50"], real_an["frames"], an["frames"]) if line == "source" else None
                d = ob.compare(an, real_an, index)
                row = {"singer": it["singer"], "sid": it["sid"], "song": it["song"], "condition": cond, "truth": TRUTH,
                       **meta, **d, "real_file": info["file"], "key_offset": info["key_offset"],
                       "path_cost": info["path_cost"]}
                for name in ("cpp_db", "hf_ratio_db", "rms_dbfs"):
                    if desc.get(name) is not None and real_desc.get(name) is not None:
                        row[f"{name}_minus_truth"] = desc[name] - real_desc[name]
                rows.append(row)
            except Exception as error:
                traceback.print_exc()
                rows.append({"singer": it["singer"], "sid": it["sid"], "condition": cond,
                             "error": f"{type(error).__name__}: {error}"})
    return rows, conditions


def main():
    from steer.adapter import TechniqueAdapter
    import torch
    if (OUT / "summary.json").exists():
        log(f"{OUT} already has summary.json; nothing to do")
        return 0
    if (OUT / "failure.json").exists():
        os.replace(OUT / "failure.json", OUT / f"failure_{int(time.time())}.json")
    manifest, items, missing = plan()
    if missing:
        raise FileNotFoundError(f"{len(missing)} GTSinger phrases missing under {DATA}, e.g. {missing[0]}")
    log(f"{len(items)} phrase pairs from {len(manifest['singers'])} GTSinger singers")
    make_refs(manifest)
    wav_dir, align_dir = OUT / "wav", OUT / "align"
    wav_dir.mkdir(parents=True, exist_ok=True)
    session = Session()
    extractor = pp.Extractor(session)
    adapter = TechniqueAdapter.from_file(session.model.cfm.estimator, g6.ADAPTER)
    adapter.off()
    started = time.perf_counter()
    stage_convert(session, extractor, adapter, items, wav_dir, align_dir)
    minutes = (time.perf_counter() - started) / 60
    log(f"conversions finished in {minutes:.1f} min; equalising")
    stage_equalise(items, wav_dir)
    log("measuring")
    rows, conditions = stage_measure(session, items, wav_dir, align_dir)
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    (OUT / "conditions.json").write_text(json.dumps(conditions, indent=0), encoding="utf-8")
    summary = {"status": "completed", "items": len(items), "rows": len(rows),
               "errors": sum("error" in r for r in rows), "convert_minutes": minutes, "results": g6.summarise(rows)}
    if torch.cuda.is_available():
        summary["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    t = summary["results"]["raw"].get(TRUTH, {})
    for name in ("real_straight", "base", "adapter_a1", "adapter_a0.5", "adapter_wrong_labels", "eqmatch",
                 "oracleeq", "static_bound", "carried"):
        if name in t:
            log(f"breathy lyric phrases, D_raw {name:20s} {t[name]['mean']:.2f}  (n = {t[name]['n']})")
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
