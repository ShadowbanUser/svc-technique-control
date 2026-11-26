"""Feasibility probe v0: is breathy phonation steerable inside frozen Seed-VC?

Uses only audio already in this project (VocalSet clips of female1/2/3):

A. determinism: same conversion twice, same seed -> noise floor
B. record:      2 sources x 4 references (f2/f3 x straight/breathy) with
                activation recording. The breathy-reference outputs double as
                the "just swap the reference" baseline.
C. directions:  difference of means (breathy - straight) per singer, plus
                agreement between the two singers' directions.
D. steer:       add singer A's direction while converting to singer B's
                NORMAL reference, sweeping strength and layer groups; with a
                random-direction control and a negative-strength control.
E. metrics:     CPP / HF ratio (breathiness proxies), F0 agreement with the
                source, CAMPPlus similarity to target and to the donor singer.

No weights are trained or modified. Outputs go to results/probe_v0/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

SOURCES = {"row": "data/vocalset/source_6s.wav",
           "dona": "data/style_pilot/source_dona_6s.wav"}
REFS = {"f2s": "data/reftech/f2_scales_straight_a_6s.wav",
        "f2b": "data/reftech/f2_scales_breathy_a_6s.wav",
        "f3s": "data/style_pilot/anchor_f3_straight_6s.wav",
        "f3b": "data/style_pilot/anchor_f3_breathy_6s.wav",
        "f3id": "data/style_pilot/identity_f3_6s.wav"}
ALPHAS = [1.0, 2.0, 4.0]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Session:
    """Loads Seed-VC once and runs the unmodified upstream conversion entry."""

    def __init__(self):
        import run_seed_baseline as base
        self.base = base
        base.configure_environment()
        manifest = json.loads((ROOT / "configs/seed_weight_manifest.json").read_text(encoding="utf-8"))
        repo, self.revision = base.verify_source(manifest)
        import numpy as np
        import torch
        self.np, self.torch = np, torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        entry, whisper, extractor = base.import_entry(repo)
        paths, _ = base.resolve_assets(manifest)
        base.adapt_offline(entry, whisper, extractor, paths)
        self.entry = entry
        original_load = entry.load_models
        cache = {}

        def load_once(args):
            if "models" not in cache:
                cache["models"] = original_load(args)
            return cache["models"]

        entry.load_models = load_once
        self.settings = dict(base.SETTINGS)
        self.seed = base.SEED
        started = time.perf_counter()
        models = entry.load_models(self._args("unused", "unused", "unused"))
        self.model, _, self.f0_fn, _, self.campplus, _, _ = models
        log(f"models loaded in {time.perf_counter() - started:.1f}s on {torch.cuda.get_device_name(0)}")
        from steer.hooks import CfmTap
        self.tap = CfmTap(self.model.cfm)

    def _args(self, source, target, output):
        return SimpleNamespace(source=str(source), target=str(target), output=str(output),
                               checkpoint=None, config=None, **self.settings)

    def convert(self, source, target, destination):
        """Run one conversion with the fixed seed; write `destination` (wav)."""
        torch = self.torch
        destination = Path(destination)
        scratch = destination.parent / "_scratch" / destination.stem
        scratch.mkdir(parents=True, exist_ok=True)
        for stale in scratch.glob("*.wav"):
            stale.unlink()
        random.seed(self.seed)
        self.np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        with torch.inference_mode():
            self.entry.main(self._args(ROOT / source, ROOT / target, scratch))
        produced = list(scratch.glob("*.wav"))
        if len(produced) != 1:
            raise RuntimeError(f"Expected one output wav in {scratch}, found {len(produced)}")
        os.replace(produced[0], destination)
        return destination

    def load16k(self, path):
        import librosa
        wave, _ = librosa.load(str(path), sr=16000)
        return wave

    def f0(self, wave_16k):
        torch = self.torch
        with torch.inference_mode():
            return self.f0_fn(torch.from_numpy(wave_16k).float().to(self.entry.device), thred=0.03)

    def embed(self, wave_16k):
        import torchaudio
        torch = self.torch
        with torch.inference_mode():
            wave = torch.from_numpy(wave_16k).float().unsqueeze(0).to(self.entry.device)
            feat = torchaudio.compliance.kaldi.fbank(wave, num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            return self.campplus(feat.unsqueeze(0))[0].float().cpu()


def cosine(a, b):
    import torch
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def run_probe(session, out):
    import torch
    from steer.hooks import difference_of_means, layer_vectors
    from steer.metrics import summarize, f0_agreement

    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    tap = session.tap

    def measure(tag, path, source_key=None, extra=None):
        row = {"tag": tag, "file": os.path.relpath(path, ROOT).replace("\\", "/")}
        row.update(extra or {})
        try:
            wave = session.load16k(path)
            f0 = session.f0(wave)
            row.update(summarize(wave, f0))
            embedding = session.embed(wave)
            for name, reference in ref_embeddings.items():
                row[f"sim_{name}"] = cosine(embedding, reference)
            if source_key is not None:
                row.update(f0_agreement(f0, source_f0[source_key]))
        except Exception as error:  # keep going; record what failed
            row["metric_error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc()
        rows.append(row)
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    # Reference material: embeddings, F0 and proxies of the real recordings.
    ref_embeddings, source_f0 = {}, {}
    for name, path in REFS.items():
        ref_embeddings[name] = session.embed(session.load16k(ROOT / path))
    for name, path in SOURCES.items():
        source_f0[name] = session.f0(session.load16k(ROOT / path))
    for name, path in {**REFS, **{f"src_{k}": v for k, v in SOURCES.items()}}.items():
        measure(f"real_{name}", ROOT / path, extra={"kind": "real"})

    # A. determinism
    log("A. determinism check")
    tap.set_steering(None)
    first = session.convert(SOURCES["row"], REFS["f3id"], wav_dir / "det_1.wav")
    second = session.convert(SOURCES["row"], REFS["f3id"], wav_dir / "det_2.wav")
    import soundfile as sf
    a, _ = sf.read(first)
    b, _ = sf.read(second)
    n = min(len(a), len(b))
    determinism = {"max_abs_diff": float(abs(a[:n] - b[:n]).max()), "length_equal": len(a) == len(b),
                   "bit_identical": bool(len(a) == len(b) and (a == b).all())}
    log(f"   {determinism}")

    # B. record
    log("B. recording activations (2 sources x 4 references)")
    recordings = {}
    for source_key, source in SOURCES.items():
        for ref_key in ("f2s", "f2b", "f3s", "f3b"):
            tag = f"rec_{source_key}_{ref_key}"
            tap.start_recording()
            path = session.convert(source, REFS[ref_key], wav_dir / f"{tag}.wav")
            recordings[source_key, ref_key] = tap.stop_recording()
            measure(tag, path, source_key, {"kind": "reference_swap", "source": source_key, "reference": ref_key})
    steps, layers, dim = recordings["row", "f2s"]["mean"].shape
    log(f"   recorded shape: steps={steps} layers={layers} dim={dim}")
    layer_sets = {"all": list(range(layers)), "early": list(range(0, layers // 3)),
                  "mid": list(range(layers // 3, 2 * layers // 3)),
                  "late": list(range(2 * layers // 3, layers))}

    # C. directions
    log("C. directions")
    directions = {}
    for singer in ("f2", "f3"):
        directions[singer] = difference_of_means(
            [recordings[s, singer + "b"] for s in SOURCES], [recordings[s, singer + "s"] for s in SOURCES])
    singer_shift = difference_of_means([recordings[s, "f3s"] for s in SOURCES],
                                       [recordings[s, "f2s"] for s in SOURCES])
    content_shift = difference_of_means([recordings["dona", r] for r in ("f2s", "f3s")],
                                        [recordings["row", r] for r in ("f2s", "f3s")])
    d2, d3 = directions["f2"]["direction"], directions["f3"]["direction"]
    geometry = {
        "per_layer_cos_f2_vs_f3_step_mean": [cosine(d2[:, l].mean(0), d3[:, l].mean(0)) for l in range(layers)],
        "per_step_cos_f2_vs_f3_layer_mean": [float(sum(cosine(d2[s, l], d3[s, l]) for l in range(layers)) / layers)
                                             for s in range(steps)],
        "per_layer_cos_technique_f2_vs_singer_shift": [cosine(d2[:, l].mean(0), singer_shift["direction"][:, l].mean(0))
                                                       for l in range(layers)],
        "per_layer_cos_technique_f2_vs_content_shift": [cosine(d2[:, l].mean(0), content_shift["direction"][:, l].mean(0))
                                                        for l in range(layers)],
        "per_layer_relative_norm_f2": directions["f2"]["relative_norm"].mean(0).tolist(),
        "per_layer_relative_norm_f3": directions["f3"]["relative_norm"].mean(0).tolist(),
        "per_layer_relative_norm_singer_shift": singer_shift["relative_norm"].mean(0).tolist(),
        "per_layer_relative_norm_content_shift": content_shift["relative_norm"].mean(0).tolist(),
        # Same-condition split (row vs dona estimate of the same f2 direction): how stable is one direction?
        "per_layer_cos_f2_row_vs_dona": [cosine(
            (recordings["row", "f2b"]["mean"] - recordings["row", "f2s"]["mean"])[:, l].mean(0),
            (recordings["dona", "f2b"]["mean"] - recordings["dona", "f2s"]["mean"])[:, l].mean(0)) for l in range(layers)],
    }
    torch.save({"f2": d2, "f3": d3, "singer_shift": singer_shift["direction"],
                "content_shift": content_shift["direction"]}, out / "directions.pt")
    (out / "geometry.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    log("   per-layer cos(f2 direction, f3 direction): " +
        " ".join(f"{v:+.2f}" for v in geometry["per_layer_cos_f2_vs_f3_step_mean"]))

    # D. steering
    log("D. steering")

    def steer_run(tag, source_key, target_key, vectors, alpha, extra):
        tap.set_steering(vectors, alpha=alpha, branch="cond")
        try:
            path = session.convert(SOURCES[source_key], REFS[target_key], wav_dir / f"{tag}.wav")
        finally:
            tap.set_steering(None)
        info = {"kind": "steer", "source": source_key, "reference": target_key, "alpha": alpha}
        info.update(extra)
        return measure(tag, path, source_key, info)

    # Random control: one random unit vector per layer, scaled to the norm of the real vector.
    generator = torch.Generator().manual_seed(session.seed)
    real_vectors = layer_vectors(d2, layer_sets["all"])
    random_vectors = {}
    for layer, vector in real_vectors.items():
        noise = torch.randn(vector.shape, generator=generator)
        random_vectors[layer] = noise / noise.norm() * vector.norm()

    for source_key in SOURCES:
        # target = f3 normal reference (a clip not used for any direction), donor direction = f2
        base_row = steer_run(f"steer_{source_key}_f3id_a0", source_key, "f3id", None, 0.0,
                             {"direction": "none", "layers": "none"})
        for set_name, layer_ids in layer_sets.items():
            vectors = layer_vectors(d2, layer_ids)
            for alpha in ALPHAS + ([8.0] if set_name == "all" else []):
                steer_run(f"steer_{source_key}_f3id_f2dir_{set_name}_a{alpha:g}", source_key, "f3id",
                          vectors, alpha, {"direction": "f2", "layers": set_name})
        steer_run(f"steer_{source_key}_f3id_f2dir_all_a-2", source_key, "f3id",
                  real_vectors, -2.0, {"direction": "f2", "layers": "all"})
        steer_run(f"steer_{source_key}_f3id_random_all_a2", source_key, "f3id",
                  random_vectors, 2.0, {"direction": "random", "layers": "all"})
        # swapped roles: target = f2 straight reference, donor direction = f3
        for alpha in ALPHAS:
            steer_run(f"steer_{source_key}_f2s_f3dir_all_a{alpha:g}", source_key, "f2s",
                      layer_vectors(d3, layer_sets["all"]), alpha, {"direction": "f3", "layers": "all"})
        log(f"   source {source_key} done (baseline cpp {base_row.get('cpp_db')})")

    summary = {"status": "completed", "seed": session.seed, "settings": session.settings,
               "source_revision": session.revision, "determinism": determinism,
               "recorded_shape": [steps, layers, dim], "runs": len(rows),
               "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
               "layer_sets": layer_sets, "alphas": ALPHAS}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"done: {len(rows)} measured items, peak GPU {summary['gpu_peak_allocated_gib']:.2f} GiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/probe_v0")
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / "summary.json").exists():
        log(f"{out} already has summary.json; nothing to do")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    stale = out / "metrics.jsonl"
    if stale.exists():
        os.replace(stale, out / f"metrics_partial_{int(time.time())}.jsonl")
    started = time.perf_counter()
    try:
        run_probe(Session(), out)
        code = 0
    except Exception as error:
        traceback.print_exc()
        (out / "failure.json").write_text(json.dumps(
            {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
            ensure_ascii=False, indent=2), encoding="utf-8")
        code = 1
    log(f"total wall {time.perf_counter() - started:.1f}s, exit {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
