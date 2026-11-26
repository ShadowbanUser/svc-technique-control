"""Experiment 03: several technique axes, several donors, and a plain-EQ baseline.

Question: does inference-time control of frozen Seed-VC do anything a static
equaliser cannot? And does averaging the contrast over several donor singers
help identity?

Axes (positive vs plain, VocalSet scales on the vowel "a"):
  breathy vs straight, belt vs straight, vocal_fry vs straight, slow_forte vs slow_piano
Donors: four singers who are neither the source nor a target.
Targets: two singers, each given one NORMAL reference only.

Arms per target x source x axis:
  cgm_w*     contrastive guidance averaged over the four donors
  cg1_w4     the same with one donor only
  stylem_w*  style-vector shift, donor-averaged
  eq_a*      static EQ: the donors' average long-term spectral difference applied
             to the uncontrolled output (no neural network involved)
  eqmatch_*  "oracle" static EQ: the long-term spectrum of the uncontrolled output is
             matched to that of a neural output. Whatever still differs between the
             two is the part of the neural change a fixed equaliser cannot make.

Reference clips are cut to 5 s and set to the same RMS before use, so loudness
differences between techniques are not transferred. Outputs: results/exp03/.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_v0 import Session, cosine, log  # noqa: E402

SR = 44100
CLIP_SECONDS = 5.0
REF_RMS_DBFS = -27.0
VOCALSET = "data/vocalset_full/FULL"
SOURCES = {"row": "data/vocalset/source_6s.wav",
           "dona": "data/style_pilot/source_dona_6s.wav"}
DONORS = ["female4", "female5", "female6", "female7"]
TARGETS = {"f3": {"existing": "data/style_pilot/identity_f3_6s.wav"},
           "f9": {"vocalset": "female9/excerpts/straight", "prefer": "caro"}}
AXES = {"breathy": ("breathy", "straight"), "belt": ("belt", "straight"),
        "fry": ("vocal_fry", "straight"), "forte": ("slow_forte", "slow_piano")}
CGM_WEIGHTS = [2.0, 4.0, 8.0, -4.0]
STYLE_WEIGHTS = [2.0, 4.0]
EQ_SCALES = [1.0, 2.0]
EQ_LIMIT_DB = 24.0
MATCH_ARMS = ["cgm_w4", "cgm_w8", "stylem_w4"]


def find_take(singer, technique):
    """Pick one VocalSet scales take for a singer/technique, preferring the vowel 'a'."""
    files = sorted(glob.glob(str(ROOT / VOCALSET / singer / "scales" / technique / "*.wav")))
    if not files:
        raise FileNotFoundError(f"No scales/{technique} take for {singer}")
    for test in (lambda n: n.endswith("_c_" + technique + "_a.wav"), lambda n: n.endswith("_a.wav")):
        chosen = [f for f in files if test(os.path.basename(f))]
        if chosen:
            return chosen[0]
    return files[0]


def cut_clip(path, seconds=CLIP_SECONDS):
    """Loudest `seconds` window of a take, mono 44.1 kHz, set to REF_RMS_DBFS (peak kept below -1 dBFS)."""
    import librosa
    import numpy as np
    wave, _ = librosa.load(str(path), sr=SR, mono=True)
    need = int(seconds * SR)
    tiled = False
    if len(wave) < need:
        wave = np.tile(wave, int(np.ceil(need / max(len(wave), 1))))[:need]
        tiled = True
    hop = SR // 10
    energy = np.cumsum(np.concatenate([[0.0], wave.astype(np.float64) ** 2]))
    starts = np.arange(0, len(wave) - need + 1, hop)
    best = int(starts[np.argmax(energy[starts + need] - energy[starts])])
    clip = wave[best:best + need].astype(np.float64)
    rms = np.sqrt(np.mean(clip ** 2)) + 1e-12
    gain = 10 ** (REF_RMS_DBFS / 20) / rms
    gain = min(gain, 10 ** (-1 / 20) / (np.max(np.abs(clip)) + 1e-12))
    return (clip * gain).astype(np.float32), {"start_seconds": best / SR, "tiled": tiled,
                                               "gain_db": float(20 * np.log10(gain))}


def third_octave_smooth(freqs, values):
    import numpy as np
    out = np.empty_like(values)
    for i, f in enumerate(freqs):
        lo, hi = f / 2 ** (1 / 6), f * 2 ** (1 / 6)
        mask = (freqs >= lo) & (freqs <= hi)
        out[i] = values[mask].mean() if mask.any() else values[i]
    return out


def ltas_db(wave, n_fft=2048, hop=512):
    import librosa
    import numpy as np
    power = np.abs(librosa.stft(wave.astype(np.float32), n_fft=n_fft, hop_length=hop)) ** 2
    return 10 * np.log10(power.mean(axis=1) + 1e-12)


def apply_eq(wave, gain_db, n_fft=2048, hop=512):
    import librosa
    import numpy as np
    spec = librosa.stft(wave.astype(np.float32), n_fft=n_fft, hop_length=hop)
    out = librosa.istft(spec * (10 ** (gain_db / 20))[:, None], hop_length=hop, length=len(wave))
    peak = np.max(np.abs(out)) + 1e-12
    return (out * min(1.0, 0.98 / peak)).astype(np.float32)


def main():
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from steer.metrics import summarize, f0_agreement, logmel_distance

    out = ROOT / "results/exp03"
    if (out / "summary.json").exists():
        log(f"{out} already has summary.json; nothing to do")
        return 0
    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = ROOT / "data/exp03_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    stale = out / "metrics.jsonl"
    if stale.exists():
        os.replace(stale, out / f"metrics_partial_{int(time.time())}.jsonl")

    # ---- reference clips -------------------------------------------------
    log("preparing reference clips")
    techniques = sorted({t for pair in AXES.values() for t in pair})
    manifest, donor_clip, donor_wave = {}, {}, {}
    for singer in DONORS:
        for technique in techniques:
            take = find_take(singer, technique)
            clip, info = cut_clip(take)
            name = f"{singer}_{technique}"
            sf.write(ref_dir / f"{name}.wav", clip, SR, subtype="PCM_16")
            donor_clip[singer, technique] = f"data/exp03_refs/{name}.wav"
            donor_wave[singer, technique] = clip
            manifest[name] = dict(source_file=os.path.relpath(take, ROOT).replace("\\", "/"), **info)
    target_clip = {}
    for name, spec in TARGETS.items():
        if "existing" in spec:
            target_clip[name] = spec["existing"]
            continue
        files = sorted(glob.glob(str(ROOT / VOCALSET / spec["vocalset"] / "*.wav")))
        chosen = [f for f in files if spec["prefer"] in os.path.basename(f)] or files
        clip, info = cut_clip(chosen[0], seconds=6.0)
        sf.write(ref_dir / f"target_{name}.wav", clip, SR, subtype="PCM_16")
        target_clip[name] = f"data/exp03_refs/target_{name}.wav"
        manifest[f"target_{name}"] = dict(source_file=os.path.relpath(chosen[0], ROOT).replace("\\", "/"), **info)
    (out / "reference_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # ---- static EQ curves (no neural network) ----------------------------
    freqs = np.fft.rfftfreq(2048, 1.0 / SR)
    eq_curve = {}
    for axis, (positive, plain) in AXES.items():
        diffs = [ltas_db(donor_wave[s, positive]) - ltas_db(donor_wave[s, plain]) for s in DONORS]
        curve = third_octave_smooth(freqs, np.mean(diffs, axis=0))
        curve[freqs < 60] = 0.0
        eq_curve[axis] = np.clip(curve, -18.0, 18.0)
    np.save(out / "eq_curves.npy", np.stack([eq_curve[a] for a in AXES]))

    session = Session()
    tap = session.tap
    embeddings = {name: session.embed(session.load16k(ROOT / path)) for name, path in target_clip.items()}
    donor_embedding = {key: session.embed(session.load16k(ROOT / path)) for key, path in donor_clip.items()}
    source_f0 = {name: session.f0(session.load16k(ROOT / path)) for name, path in SOURCES.items()}
    rows, base_wave = [], {}

    def reset():
        tap.set_steering(None)
        tap.set_guidance(None)
        tap.set_style_delta(None)

    def measure(tag, path, target, source_key, info):
        row = {"tag": tag, "file": os.path.relpath(path, ROOT).replace("\\", "/"),
               "target": target, "source": source_key}
        row.update(info)
        try:
            wave = session.load16k(path)
            f0 = session.f0(wave)
            row.update(summarize(wave, f0))
            row.update(f0_agreement(f0, source_f0[source_key]))
            embedding = session.embed(wave)
            row["sim_target"] = cosine(embedding, embeddings[target])
            axis = info.get("axis")
            if axis:
                positive = AXES[axis][0]
                row["sim_donors_mean"] = float(np.mean([cosine(embedding, donor_embedding[s, positive]) for s in DONORS]))
            key = (target, source_key)
            if info.get("arm") == "base":
                base_wave[key] = wave
            row["logmel_change_db"] = logmel_distance(wave, base_wave[key]) if key in base_wave else None
        except Exception as error:
            row["metric_error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc()
        rows.append(row)
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    # ---- conditioning packs ---------------------------------------------
    log("capturing conditioning packs (2 sources x 4 donors x 6 clips)")
    packs = {}
    scratch = wav_dir / "_packs"
    scratch.mkdir(exist_ok=True)
    for source_key, source in SOURCES.items():
        for (singer, technique), clip in donor_clip.items():
            tap.start_capture()
            session.convert(source, clip, scratch / f"{source_key}_{singer}_{technique}.wav")
            packs[source_key, singer, technique] = tap.stop_capture()

    # ---- arms ------------------------------------------------------------
    for target, reference in target_clip.items():
        for source_key, source in SOURCES.items():
            log(f"target {target} / source {source_key}")

            def run(arm, info):
                tag = f"{target}_{source_key}_{arm}"
                started = time.perf_counter()
                try:
                    path = session.convert(source, reference, wav_dir / f"{tag}.wav")
                finally:
                    reset()
                return measure(tag, path, target, source_key,
                               dict(arm=arm, seconds=time.perf_counter() - started, **info))

            run("base", {"method": "none", "strength": 0.0, "axis": None})
            base_path = wav_dir / f"{target}_{source_key}_base.wav"
            base_44k, _ = librosa.load(str(base_path), sr=SR, mono=True)
            for axis, (positive, plain) in AXES.items():
                pos = [packs[source_key, s, positive] for s in DONORS]
                neg = [packs[source_key, s, plain] for s in DONORS]
                for weight in CGM_WEIGHTS:
                    tap.set_guidance(pos, neg, weight)
                    run(f"{axis}_cgm_w{weight:g}", {"method": "guidance_multi", "strength": weight, "axis": axis})
                tap.set_guidance(pos[0], neg[0], 4.0)
                run(f"{axis}_cg1_w4", {"method": "guidance_single", "strength": 4.0, "axis": axis})
                delta = torch.stack([p["style"] - n["style"] for p, n in zip(pos, neg)]).mean(0)
                for weight in STYLE_WEIGHTS:
                    tap.set_style_delta(weight * delta)
                    run(f"{axis}_stylem_w{weight:g}", {"method": "style_multi", "strength": weight, "axis": axis})
                for scale in EQ_SCALES:
                    tag = f"{target}_{source_key}_{axis}_eq_a{scale:g}"
                    path = wav_dir / f"{tag}.wav"
                    gain = np.clip(scale * eq_curve[axis], -EQ_LIMIT_DB, EQ_LIMIT_DB)
                    sf.write(path, apply_eq(base_44k, gain), SR, subtype="PCM_16")
                    measure(tag, path, target, source_key,
                            {"arm": f"{axis}_eq_a{scale:g}", "method": "static_eq", "strength": scale, "axis": axis})
                for arm in MATCH_ARMS:
                    neural_path = wav_dir / f"{target}_{source_key}_{axis}_{arm}.wav"
                    neural_44k, _ = librosa.load(str(neural_path), sr=SR, mono=True)
                    gain = third_octave_smooth(freqs, ltas_db(neural_44k) - ltas_db(base_44k))
                    gain[freqs < 60] = 0.0
                    gain = np.clip(gain, -EQ_LIMIT_DB, EQ_LIMIT_DB)
                    tag = f"{target}_{source_key}_{axis}_eqmatch_{arm}"
                    path = wav_dir / f"{tag}.wav"
                    sf.write(path, apply_eq(base_44k, gain), SR, subtype="PCM_16")
                    row = measure(tag, path, target, source_key,
                                  {"arm": f"{axis}_eqmatch_{arm}", "method": "matched_eq", "strength": None,
                                   "axis": axis, "matches": f"{axis}_{arm}"})
                    try:
                        neural_16k, matched_16k = session.load16k(neural_path), session.load16k(path)
                        row["neural_change_db"] = logmel_distance(neural_16k, base_wave[target, source_key])
                        row["residual_after_eq_db"] = logmel_distance(neural_16k, matched_16k)
                    except Exception as error:
                        row["residual_error"] = f"{type(error).__name__}: {error}"

    summary = {"status": "completed", "seed": session.seed, "settings": session.settings, "runs": len(rows),
               "donors": DONORS, "targets": target_clip, "axes": {k: list(v) for k, v in AXES.items()},
               "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
    (out / "metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"done: {len(rows)} measured items, peak GPU {summary['gpu_peak_allocated_gib']:.2f} GiB")
    return 0


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        code = main()
    except Exception as error:
        traceback.print_exc()
        failure = ROOT / "results/exp03"
        failure.mkdir(parents=True, exist_ok=True)
        (failure / "failure.json").write_text(json.dumps(
            {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
            ensure_ascii=False, indent=2), encoding="utf-8")
        code = 1
    log(f"total wall {time.perf_counter() - started:.1f}s, exit {code}")
    raise SystemExit(code)
