"""Run the unmodified, locked Seed-VC v1 SVC entry point using offline assets."""
import argparse
import contextlib
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(f0_condition=True, auto_f0_adjust=False, semi_tone_shift=0,
                length_adjust=1.0, diffusion_steps=30, inference_cfg_rate=0.7,
                fp16=True)
SEED = 20261020


def digest(path, algorithm="sha256", git_blob=False):
    result = hashlib.new(algorithm)
    if git_blob:
        result.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def configure_environment():
    sys.dont_write_bytecode = True
    cache = ROOT / "checkpoints"
    for name, path in {"HF_HOME": cache / "huggingface",
                       "HF_HUB_CACHE": cache / "huggingface" / "hub",
                       "TORCH_HOME": cache / "torch",
                       "NUMBA_CACHE_DIR": cache / "numba",
                       "MPLCONFIGDIR": cache / "matplotlib",
                       "XDG_CACHE_HOME": cache / "xdg",
                       "CUDA_CACHE_PATH": cache / "cuda",
                       "JOBLIB_TEMP_FOLDER": cache / "tmp-seed",
                       "TMP": cache / "tmp-seed", "TEMP": cache / "tmp-seed"}.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1")


def verify_source(manifest):
    repo = ROOT / "external" / "seed-vc"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                       text=True).strip()
    if revision != manifest["source_revision"]:
        raise ValueError(f"Upstream source revision mismatch: {revision}")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=repo,
                   check=True, capture_output=True)
    return repo, revision


def resolve_assets(manifest):
    from huggingface_hub import try_to_load_from_cache
    paths, checked = {}, []
    for item in manifest["weights"] + manifest["configs"]:
        cached = try_to_load_from_cache(item["repo_id"], item["filename"],
                    revision=item["revision"], cache_dir=os.environ["HF_HUB_CACHE"])
        if not isinstance(cached, str) or not Path(cached).is_file():
            raise FileNotFoundError(f"Pinned asset is missing: {item['repo_id']}@"
                                    f"{item['revision']}/{item['filename']}")
        path = Path(cached)
        if path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"Asset size mismatch: {path}")
        expected = item.get("sha256_lfs")
        actual = digest(path) if expected else digest(path, "sha1", git_blob=True)
        if actual != (expected or item["git_blob_id"]):
            raise ValueError(f"Asset checksum mismatch: {path}")
        paths[item["repo_id"], item["filename"]] = path
        checked.append(dict(repo_id=item["repo_id"], revision=item["revision"],
                            filename=item["filename"], path=str(path),
                            size_bytes=item["size_bytes"], verified_checksum=actual))
    return paths, checked


def authorize_audio(path, authorizations):
    path = path.resolve(strict=True)
    forbidden = (ROOT / "external" / "seed-vc" / "examples").resolve()
    if path == forbidden or forbidden in path.parents:
        raise ValueError("Upstream example audio is not an authorized experiment input")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Baseline inputs must be WAV: {path}")
    checksum = digest(path)
    for item in authorizations["audio"]:
        candidate = Path(item["path"])
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.resolve() != path:
            continue
        if (item.get("authorized_for_research") is not True or
                item.get("sha256") != checksum or not item.get("provenance") or
                not item.get("authorization")):
            raise ValueError(f"Incomplete or mismatched audio authorization: {path}")
        import numpy as np
        import soundfile as sf
        samples, sr = sf.read(path, always_2d=True)
        if not samples.size or sr <= 0 or not np.isfinite(samples).all():
            raise ValueError(f"Empty or nonfinite audio: {path}")
        return dict(path=str(path), sha256=checksum, sample_rate=sr,
                    channels=samples.shape[1], duration_seconds=len(samples) / sr,
                    provenance=item["provenance"], authorization=item["authorization"])
    raise ValueError(f"No matching research authorization for {path}")


def import_entry(repo):
    sys.path.insert(0, str(repo))
    # Import every module on the selected v1 SVC path without loading weights.
    for name in ("modules.flow_matching", "modules.length_regulator", "modules.rmvpe",
                 "modules.campplus.DTDNN", "modules.bigvgan.bigvgan", "modules.audio"):
        importlib.import_module(name)
    from transformers import WhisperModel, AutoFeatureExtractor
    spec = importlib.util.spec_from_file_location("locked_seed_inference", repo / "inference.py")
    entry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = entry
    spec.loader.exec_module(entry)
    # Upstream overwrites HF_HUB_CACHE on import; restore the project cache.
    configure_environment()
    return entry, WhisperModel, AutoFeatureExtractor


def adapt_offline(entry, whisper, extractor, paths):
    def local_loader(repo_id, model_filename="pytorch_model.bin", config_filename=None):
        model = str(paths[repo_id, model_filename])
        return (model, str(paths[repo_id, config_filename])) if config_filename else model

    def redirect(cls, repo_id, local_dir):
        original = cls.from_pretrained

        def from_local(_cls, name, *args, **kwargs):
            if name != repo_id:
                raise ValueError(f"Unapproved pretrained model: {name}")
            kwargs["local_files_only"] = True
            return original(str(local_dir), *args, **kwargs)

        cls.from_pretrained = classmethod(from_local)

    entry.load_custom_model_from_hf = local_loader
    whisper_dir = paths["openai/whisper-small", "model.safetensors"].parent
    redirect(whisper, "openai/whisper-small", whisper_dir)
    redirect(extractor, "openai/whisper-small", whisper_dir)
    from modules.bigvgan.bigvgan import BigVGAN
    redirect(BigVGAN, "nvidia/bigvgan_v2_44khz_128band_512x",
             paths["nvidia/bigvgan_v2_44khz_128band_512x", "bigvgan_generator.pt"].parent)

    # Torch 2.8 audio export may require TorchCodec. PCM WAV needs only soundfile.
    import soundfile as sf

    def save_pcm(path, samples, sample_rate, **kwargs):
        if kwargs:
            raise ValueError(f"Unexpected upstream audio export options: {kwargs}")
        sf.write(path, samples.detach().cpu().numpy().T, sample_rate, subtype="PCM_16")

    entry.torchaudio.save = save_pcm


def run(args, record):
    configure_environment()
    if args.self_check:
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory(dir=os.environ["TMP"]) as folder:
            path = Path(folder) / "authorization_test.wav"
            sf.write(path, np.sin(np.arange(1600) * 0.1) * 0.1, 16000)
            item = dict(path=str(path), sha256=digest(path), authorized_for_research=True,
                        provenance="Synthetic utility check", authorization="Generated for this check")
            result = authorize_audio(path, {"audio": [item]})
            assert result["duration_seconds"] == 0.1
            for change in ({"sha256": "wrong"}, {"authorized_for_research": False}):
                try:
                    authorize_audio(path, {"audio": [{**item, **change}]})
                except ValueError:
                    pass
                else:
                    raise AssertionError("Invalid authorization was accepted")
        record["status"] = "self_check_passed_synthetic_utility_only"
        return
    manifest_path = args.weight_manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo, revision = verify_source(manifest)
    record.update(source_revision=revision, manifest=str(manifest_path),
                  manifest_sha256=digest(manifest_path), seed=SEED, settings=SETTINGS)
    import torch
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this measured baseline")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    record.update(python=sys.version, torch=torch.__version__,
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  determinism="Seed fixed; exact CUDA bitwise determinism is not guaranteed")
    entry, whisper, extractor = import_entry(repo)
    record["entry_imported_without_weights"] = True
    if args.preflight:
        record["status"] = "preflight_passed_no_weights_loaded_no_audio_inference"
        return
    paths, record["assets"] = resolve_assets(manifest)
    if args.check_cache:
        record["status"] = "pinned_cache_verified_no_weights_loaded"
        return
    if args.source is None or args.target is None:
        raise ValueError("--source and --target are required for audio inference")
    authorization_path = args.authorization_manifest.resolve(strict=True)
    authorizations = json.loads(authorization_path.read_text(encoding="utf-8"))
    record["authorization_manifest"] = str(authorization_path)
    record["authorization_manifest_sha256"] = digest(authorization_path)
    source = authorize_audio(args.source, authorizations)
    target = authorize_audio(args.target, authorizations)
    record.update(source=source, target=target)
    adapt_offline(entry, whisper, extractor, paths)
    svc = SimpleNamespace(source=source["path"], target=target["path"],
        output=str(args.output_dir), checkpoint=None, config=None, **SETTINGS)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    measured_start = time.perf_counter()
    try:
        with torch.inference_mode():
            entry.main(svc)
        torch.cuda.synchronize()
    finally:
        record.update(model_load_and_inference_wall_seconds=time.perf_counter() - measured_start,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    import soundfile as sf
    output = args.output_dir / (f"vc_{Path(source['path']).name.split('.')[0]}_"
                              f"{Path(target['path']).name.split('.')[0]}_1.0_30_0.7.wav")
    audio, sr = sf.read(output, always_2d=True)
    if sr != 44100 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Invalid output audio or unexpected sample rate")
    duration = len(audio) / sr
    record.update(status="completed", output=str(output), output_sha256=digest(output),
                  output_sample_rate=sr, output_channels=audio.shape[1],
                  output_duration_seconds=duration, output_peak=float(np.abs(audio).max()),
                  output_rms=float(np.sqrt(np.mean(audio ** 2))),
                  output_clipped_sample_fraction=float(np.mean(np.abs(audio) >= 32767 / 32768)),
                  rtf_including_model_load=record["model_load_and_inference_wall_seconds"] / duration,
                  rtf_definition="CUDA-synchronized load+inference wall seconds / output seconds; excludes hash verification and import; upstream printed RTF has its own narrower scope")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="Import dependencies only; load no weights")
    mode.add_argument("--check-cache", action="store_true", help="Verify pinned cache hashes; load no weights")
    mode.add_argument("--self-check", action="store_true", help="Test audio authorization guards with a synthetic signal")
    parser.add_argument("--weight-manifest", type=Path, default=ROOT / "configs/seed_weight_manifest.json")
    parser.add_argument("--authorization-manifest", type=Path, default=ROOT / "configs/audio_authorizations.json")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    default = ROOT / ("logs" if args.preflight or args.check_cache or args.self_check else "results")
    args.output_dir = (args.output_dir or default / f"seed_setup_{stamp}").resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(start_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  command=[sys.executable, *sys.argv], status="running", offline=True,
                  adapter_changes=["local pinned HF assets", "soundfile PCM16 export"])
    started = time.perf_counter()
    code = 0
    with (args.output_dir / "stdout.log").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)), contextlib.redirect_stderr(Tee(sys.stderr, stream)):
            try:
                run(args, record)
            except Exception as error:
                code = 1
                record.update(status="failed", error_type=type(error).__name__, error=str(error))
                traceback.print_exc()
            finally:
                record.update(end_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                              total_wall_seconds=time.perf_counter() - started)
                (args.output_dir / "run.json").write_text(json.dumps(record, ensure_ascii=False,
                    indent=2) + "\n", encoding="utf-8")
                print(json.dumps(record, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
