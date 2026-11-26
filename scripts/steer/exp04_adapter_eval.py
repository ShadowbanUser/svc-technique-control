"""Experiment 04 evaluation: the trained technique adapter on the exp03 test cells.

Cells: targets f3, f9 (one normal reference each, both singers unseen in training)
x sources row, dona. Arms per cell:

  base                        adapter off; must equal exp03's base output bit for bit
  <tech>_a{0.5,1,1.5}         one technique at a time
  combo_breathy_fry           breathy 0.7 + vocal_fry 0.7
  combo_breathy_piano         breathy 1.0 + piano 1.0
  <tech>_eqmatch_a1           plain EQ matched to the long-term spectrum of <tech>_a1

Outputs: results/exp04/eval/
"""
from __future__ import annotations

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
from exp03_panel import SR, SOURCES, EQ_LIMIT_DB, ltas_db, third_octave_smooth, apply_eq  # noqa: E402

ADAPTER = ROOT / "results/exp04/train/adapter_best.pt"
OUT = ROOT / "results/exp04/eval"
TARGETS = {"f3": "data/style_pilot/identity_f3_6s.wav", "f9": "data/exp03_refs/target_f9.wav"}
AMOUNTS = [0.5, 1.0, 1.5]
COMBOS = {"combo_breathy_fry": {"breathy": 0.7, "vocal_fry": 0.7},
          "combo_breathy_piano": {"breathy": 1.0, "piano": 1.0}}


def main():
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from steer.adapter import TechniqueAdapter
    from steer.metrics import summarize, f0_agreement, logmel_distance

    if (OUT / "summary.json").exists():
        log(f"{OUT} already has summary.json; nothing to do")
        return 0
    if not ADAPTER.exists():
        raise FileNotFoundError(f"{ADAPTER} not found; training has not finished")
    wav_dir = OUT / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    stale = OUT / "metrics.jsonl"
    if stale.exists():
        os.replace(stale, OUT / f"metrics_partial_{int(time.time())}.jsonl")

    session = Session()
    session.tap.close()
    adapter = TechniqueAdapter.from_file(session.model.cfm.estimator, ADAPTER)
    adapter.off()
    techniques = adapter.techniques
    log(f"adapter loaded: {techniques}, rank {adapter.rank}")

    embeddings = {k: session.embed(session.load16k(ROOT / v)) for k, v in TARGETS.items()}
    source_f0 = {k: session.f0(session.load16k(ROOT / v)) for k, v in SOURCES.items()}
    freqs = np.fft.rfftfreq(2048, 1.0 / SR)
    rows, base16 = [], {}

    def measure(tag, path, target, source_key, info):
        row = {"tag": tag, "file": os.path.relpath(path, ROOT).replace("\\", "/"),
               "target": target, "source": source_key, **info}
        try:
            wave = session.load16k(path)
            f0 = session.f0(wave)
            row.update(summarize(wave, f0))
            row.update(f0_agreement(f0, source_f0[source_key]))
            row["sim_target"] = cosine(session.embed(wave), embeddings[target])
            key = (target, source_key)
            if info["arm"] == "base":
                base16[key] = wave
            row["logmel_change_db"] = logmel_distance(wave, base16[key]) if key in base16 else None
        except Exception as error:
            row["metric_error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc()
        rows.append(row)
        with (OUT / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    checks = {}
    for target, reference in TARGETS.items():
        for source_key, source in SOURCES.items():
            log(f"target {target} / source {source_key}")

            def run(arm, amounts, extra):
                tag = f"{target}_{source_key}_{arm}"
                adapter.set(**amounts) if amounts else adapter.off()
                started = time.perf_counter()
                try:
                    path = session.convert(source, reference, wav_dir / f"{tag}.wav")
                finally:
                    adapter.off()
                return measure(tag, path, target, source_key,
                               {"arm": arm, "amounts": amounts, "seconds": time.perf_counter() - started, **extra})

            run("base", {}, {"method": "none"})
            previous = ROOT / f"results/exp03/wav/{target}_{source_key}_base.wav"
            if previous.exists():
                a, _ = sf.read(str(wav_dir / f"{target}_{source_key}_base.wav"), dtype="int16")
                b, _ = sf.read(str(previous), dtype="int16")
                checks[f"{target}_{source_key}"] = bool(a.shape == b.shape and np.array_equal(a, b))
            base44, _ = librosa.load(str(wav_dir / f"{target}_{source_key}_base.wav"), sr=SR, mono=True)
            for tech in techniques:
                for amount in AMOUNTS:
                    run(f"{tech}_a{amount:g}", {tech: amount}, {"method": "adapter", "axis": tech, "strength": amount})
                if 1.0 not in AMOUNTS:
                    continue
                neural = wav_dir / f"{target}_{source_key}_{tech}_a1.wav"
                neural44, _ = librosa.load(str(neural), sr=SR, mono=True)
                gain = third_octave_smooth(freqs, ltas_db(neural44) - ltas_db(base44))
                gain[freqs < 60] = 0.0
                gain = np.clip(gain, -EQ_LIMIT_DB, EQ_LIMIT_DB)
                tag = f"{target}_{source_key}_{tech}_eqmatch_a1"
                path = wav_dir / f"{tag}.wav"
                sf.write(path, apply_eq(base44, gain), SR, subtype="PCM_16")
                row = measure(tag, path, target, source_key,
                              {"arm": f"{tech}_eqmatch_a1", "method": "matched_eq", "axis": tech, "matches": f"{tech}_a1"})
                try:
                    neural16, matched16 = session.load16k(neural), session.load16k(path)
                    row["neural_change_db"] = logmel_distance(neural16, base16[target, source_key])
                    row["residual_after_eq_db"] = logmel_distance(neural16, matched16)
                except Exception as error:
                    row["residual_error"] = f"{type(error).__name__}: {error}"
            for arm, amounts in COMBOS.items():
                run(arm, amounts, {"method": "adapter_combo"})

    summary = {"status": "completed", "adapter": str(ADAPTER.relative_to(ROOT)), "techniques": techniques,
               "runs": len(rows), "base_matches_exp03": checks, "seed": session.seed, "settings": session.settings,
               "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
    (OUT / "metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"done: {len(rows)} items; base identical to exp03: {checks}")
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
