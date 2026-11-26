"""Experiment 02: three ways to push a frozen Seed-VC conversion toward breathy phonation.

Cross-singer setting throughout: the target singer is given only a NORMAL
reference; the breathy/plain contrast comes from a different (donor) singer.

Arms, per target singer and source clip:
  base            no control
  own_breathy     reference = the target singer's own breathy clip (what reference
                  swapping gives when such a clip exists; a ceiling, not a method)
  cg_w*           contrastive guidance: velocity += w * (v(donor breathy) - v(donor plain))
  style_w*        style vector += w * (style(donor breathy) - style(donor plain))
  act_*           activation addition with per-step vectors on mid layers / single layers
  rand_*          the same with a random direction of matched norm (control)

Outputs: results/exp02/. No weights are trained or modified.
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

SOURCES = {"row": "data/vocalset/source_6s.wav",
           "dona": "data/style_pilot/source_dona_6s.wav"}
CLIPS = {"f2s": "data/reftech/f2_scales_straight_a_6s.wav",
         "f2b": "data/reftech/f2_scales_breathy_a_6s.wav",
         "f3s": "data/style_pilot/anchor_f3_straight_6s.wav",
         "f3b": "data/style_pilot/anchor_f3_breathy_6s.wav",
         "f3id": "data/style_pilot/identity_f3_6s.wav",
         "f2row": "data/vocalset/target_6s.wav"}
# target normal reference, donor plain, donor breathy, target's own breathy clip, donor direction key
SETTINGS = {"toF3": dict(target="f3id", plain="f2s", breathy="f2b", own="f3b", direction="f2"),
            "toF2": dict(target="f2row", plain="f3s", breathy="f3b", own="f2b", direction="f3")}
CG_WEIGHTS = [0.5, 1.0, 2.0, 4.0, -1.0, -2.0]
STYLE_WEIGHTS = [1.0, 2.0, 4.0]
MID_ALPHAS = [4.0, 8.0, 16.0]
SINGLE_LAYERS = [6, 8, 10]
SINGLE_ALPHAS = [8.0, 16.0]


def main():
    import torch
    from steer.hooks import layer_vectors
    from steer.metrics import summarize, f0_agreement, logmel_distance

    out = ROOT / "results/exp02"
    if (out / "summary.json").exists():
        log(f"{out} already has summary.json; nothing to do")
        return 0
    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    stale = out / "metrics.jsonl"
    if stale.exists():
        os.replace(stale, out / f"metrics_partial_{int(time.time())}.jsonl")

    session = Session()
    tap = session.tap
    directions = torch.load(ROOT / "results/probe_v0/directions.pt", map_location="cpu")
    steps, layers, _ = directions["f2"].shape
    mid = list(range(layers // 3, 2 * layers // 3))

    embeddings = {name: session.embed(session.load16k(ROOT / path)) for name, path in CLIPS.items()}
    source_f0 = {name: session.f0(session.load16k(ROOT / path)) for name, path in SOURCES.items()}
    rows, base_wave = [], {}

    def measure(tag, path, setting, source_key, info):
        row = {"tag": tag, "file": os.path.relpath(path, ROOT).replace("\\", "/"),
               "setting": setting, "source": source_key}
        row.update(info)
        try:
            wave = session.load16k(path)
            f0 = session.f0(wave)
            row.update(summarize(wave, f0))
            row.update(f0_agreement(f0, source_f0[source_key]))
            embedding = session.embed(wave)
            cfg = SETTINGS[setting]
            row["sim_target"] = cosine(embedding, embeddings[cfg["target"]])
            row["sim_target_own_breathy"] = cosine(embedding, embeddings[cfg["own"]])
            row["sim_donor_plain"] = cosine(embedding, embeddings[cfg["plain"]])
            row["sim_donor_breathy"] = cosine(embedding, embeddings[cfg["breathy"]])
            key = (setting, source_key)
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

    def reset():
        tap.set_steering(None)
        tap.set_guidance(None)
        tap.set_style_delta(None)

    # Conditioning packs: each source converted to each donor clip, capturing the solver inputs.
    log("capturing conditioning packs")
    packs = {}
    for source_key, source in SOURCES.items():
        for clip in ("f2s", "f2b", "f3s", "f3b"):
            tap.start_capture()
            session.convert(source, CLIPS[clip], wav_dir / f"pack_{source_key}_{clip}.wav")
            packs[source_key, clip] = tap.stop_capture()

    # Sanity on real weights: same singer, weight 1 + cfg should reproduce the reference swap.
    log("sanity: guidance with weight 1+cfg vs reference swap (same singer)")
    cfg_rate = session.settings["inference_cfg_rate"]
    tap.set_guidance(packs["row", "f2b"], packs["row", "f2s"], 1.0 + cfg_rate)
    try:
        guided = session.convert(SOURCES["row"], CLIPS["f2s"], wav_dir / "sanity_row_f2s_guided.wav")
    finally:
        reset()
    a = session.load16k(guided)
    b = session.load16k(wav_dir / "pack_row_f2b.wav")
    c = session.load16k(wav_dir / "pack_row_f2s.wav")
    sanity = {"logmel_guided_vs_breathy_ref_db": logmel_distance(a, b),
              "logmel_plain_ref_vs_breathy_ref_db": logmel_distance(c, b)}
    log(f"   {sanity}")

    for setting, cfg in SETTINGS.items():
        direction = directions[cfg["direction"]]
        generator = torch.Generator().manual_seed(session.seed)
        random_direction = torch.zeros_like(direction)
        for layer in range(layers):
            unit = torch.randn(direction.shape[-1], generator=generator)
            unit = unit / unit.norm()
            random_direction[:, layer] = unit[None, :] * direction[:, layer].norm(dim=-1, keepdim=True)
        for source_key, source in SOURCES.items():
            log(f"{setting} / {source_key}")
            target = CLIPS[cfg["target"]]

            def run(arm, info, reference=target):
                tag = f"{setting}_{source_key}_{arm}"
                try:
                    path = session.convert(source, reference, wav_dir / f"{tag}.wav")
                finally:
                    reset()
                return measure(tag, path, setting, source_key, dict(arm=arm, **info))

            run("base", {"method": "none", "strength": 0.0})
            run("own_breathy", {"method": "reference_swap", "strength": None}, reference=CLIPS[cfg["own"]])
            for weight in CG_WEIGHTS:
                tap.set_guidance(packs[source_key, cfg["breathy"]], packs[source_key, cfg["plain"]], weight)
                run(f"cg_w{weight:g}", {"method": "guidance", "strength": weight})
            delta = packs[source_key, cfg["breathy"]]["style"] - packs[source_key, cfg["plain"]]["style"]
            for weight in STYLE_WEIGHTS:
                tap.set_style_delta(weight * delta)
                run(f"style_w{weight:g}", {"method": "style_vector", "strength": weight})
            for alpha in MID_ALPHAS:
                tap.set_steering(layer_vectors(direction, mid, per_step=True), alpha=alpha)
                run(f"act_mid_a{alpha:g}", {"method": "activation_mid", "strength": alpha})
            for layer in SINGLE_LAYERS:
                for alpha in SINGLE_ALPHAS:
                    tap.set_steering(layer_vectors(direction, [layer], per_step=True), alpha=alpha)
                    run(f"act_L{layer}_a{alpha:g}", {"method": f"activation_L{layer}", "strength": alpha})
            for alpha in (8.0, 16.0):
                tap.set_steering(layer_vectors(random_direction, mid, per_step=True), alpha=alpha)
                run(f"rand_mid_a{alpha:g}", {"method": "random_mid", "strength": alpha})

    summary = {"status": "completed", "seed": session.seed, "settings": session.settings,
               "runs": len(rows), "sanity": sanity, "mid_layers": mid,
               "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
    (out / "metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"done: {len(rows)} runs, peak GPU {summary['gpu_peak_allocated_gib']:.2f} GiB")
    return 0


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        code = main()
    except Exception as error:
        traceback.print_exc()
        failure = ROOT / "results/exp02"
        failure.mkdir(parents=True, exist_ok=True)
        (failure / "failure.json").write_text(json.dumps(
            {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
            ensure_ascii=False, indent=2), encoding="utf-8")
        code = 1
    log(f"total wall {time.perf_counter() - started:.1f}s, exit {code}")
    raise SystemExit(code)
