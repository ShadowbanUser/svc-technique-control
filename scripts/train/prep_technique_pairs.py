"""Build training pairs for the technique adapter from VocalSet.

For every singer, set (scales / arpeggios) and vowel, the straight take is the
content source and each technique take of the same exercise is the target:

  target mel     : the technique take (e.g. breathy), 44.1 kHz log-mel
  target content : Whisper features of the STRAIGHT take, time-warped onto the
                   technique take with DTW (pitch contour + voicing + energy)
  target F0      : F0 of the STRAIGHT take, warped the same way
  reference      : another straight take of the same singer (chosen at training time)

So at training time the technique can only come from the label, exactly as at
inference time, where the source is ordinary singing and the reference is the
target singer's ordinary voice.

Output: cache/techpairs/<singer>.pt and cache/techpairs/manifest.json
"""
from __future__ import annotations

import json
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/steer"))
sys.path.insert(0, str(ROOT / "src"))

VOCALSET = ROOT / "data/vocalset_full/FULL"
OUT = ROOT / "cache/techpairs"
SETS = ("scales", "arpeggios")
TECHNIQUES = {"breathy": "breathy", "belt": "belt", "vocal_fry": "vocal_fry",
              "slow_forte": "forte", "slow_piano": "piano"}
VOWEL = re.compile(r"_([aeiou])(\(\d\))?\.wav$")
SR = 44100
TRIM_DB = 35
PAD_SECONDS = 0.15


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def takes(folder):
    """vowel -> list of files (files with a '(1)' duplicate marker go last)."""
    found = {}
    for path in sorted(folder.glob("*.wav")):
        match = VOWEL.search(path.name)
        if match:
            found.setdefault(match.group(1), []).append(path)
    for vowel in found:
        found[vowel].sort(key=lambda p: ("(" in p.name, p.name))
    return found


def load_trimmed(path):
    import librosa
    import numpy as np
    wave, _ = librosa.load(str(path), sr=SR, mono=True)
    _, (start, end) = librosa.effects.trim(wave, top_db=TRIM_DB, frame_length=2048, hop_length=512)
    pad = int(PAD_SECONDS * SR)
    wave = wave[max(0, start - pad):min(len(wave), end + pad)]
    return np.ascontiguousarray(wave, dtype=np.float32)


def semitones(f0):
    import numpy as np
    voiced = f0 > 1
    st = np.zeros_like(f0, dtype=np.float64)
    st[voiced] = 12 * np.log2(f0[voiced] / 440.0)
    return st, voiced


def log_energy(w16, n):
    import numpy as np
    hop, win = 320, 640
    padded = np.pad(w16, (win // 2, win // 2))
    frames = [padded[i * hop:i * hop + win] for i in range(n)]
    frames = [f if len(f) == win else np.pad(f, (0, win - len(f))) for f in frames]
    e = np.log10(np.mean(np.square(np.stack(frames)), axis=1) + 1e-8)
    return (e - e.mean()) / (e.std() + 1e-6)


def note_offset(st_s, v_s, st_t, v_t, min_frames=3):
    """Integer key offset (semitones) between two takes of the same exercise.

    Compares which notes occur, not how long each is held, so differences in
    tempo or note length do not bias it. Returns (offset, overlap score)."""
    import numpy as np

    def notes(st, voiced):
        values, counts = np.unique(np.round(st[voiced]).astype(int), return_counts=True)
        return set(values[counts >= min_frames].tolist())
    a, b = notes(st_s, v_s), notes(st_t, v_t)
    if not a or not b:
        return 0, 0.0
    best = (-1.0, 0)
    for offset in range(-14, 15):
        shifted = {n - offset for n in b}
        score = len(a & shifted) / len(a | shifted)
        if score > best[0] + 1e-9 or (abs(score - best[0]) <= 1e-9 and abs(offset) < abs(best[1])):
            best = (score, offset)
    return best[1], best[0]


def align(src, tgt, pitch_weight):
    """DTW on 50 Hz frames. Returns map (target frame -> source frame) and diagnostics."""
    import librosa
    import numpy as np
    st_s, v_s = src["st50"], src["v50"]
    st_t, v_t = tgt["st50"], tgt["v50"]
    offset, overlap = note_offset(st_s, v_s, st_t, v_t)
    if overlap >= 0.5:
        norm_s, norm_t = st_s, st_t - offset
    else:  # pitch sets disagree (e.g. unreliable F0): fall back to median normalisation, no key shift
        offset = 0
        norm_s = st_s - (np.median(st_s[v_s]) if v_s.any() else 0.0)
        norm_t = st_t - (np.median(st_t[v_t]) if v_t.any() else 0.0)
    both = v_s[:, None] & v_t[None, :]
    pitch = np.minimum(np.abs(norm_s[:, None] - norm_t[None, :]), 4.0) / 4.0
    cost = (pitch_weight * np.where(both, pitch, 0.0)
            + 0.7 * (v_s[:, None] != v_t[None, :])
            + 0.3 * np.minimum(np.abs(src["e50"][:, None] - tgt["e50"][None, :]), 3.0) / 3.0)
    try:
        acc, path = librosa.sequence.dtw(C=cost, global_constraints=True, band_rad=0.3)
        if not np.isfinite(acc[-1, -1]):
            raise ValueError("band too narrow")
    except Exception:
        acc, path = librosa.sequence.dtw(C=cost)
    path = path[::-1]
    mapping = np.zeros(cost.shape[1], dtype=np.int64)
    for j in range(cost.shape[1]):
        mapping[j] = int(round(path[path[:, 1] == j, 0].mean()))
    path_cost = float(cost[path[:, 0], path[:, 1]].mean())
    voiced_match = float((v_s[mapping] == v_t).mean())
    both = v_t & v_s[mapping]
    residual = st_t[both] - st_s[mapping][both] - offset
    pitch_residual = float(np.median(np.abs(residual))) if residual.size else None
    return mapping, {"path_cost": path_cost, "voiced_match": voiced_match, "key_offset": int(offset),
                     "note_overlap": float(overlap), "pitch_residual_semitones": pitch_residual}


class Extractor:
    def __init__(self, session):
        import torch
        self.torch = torch
        self.device = session.entry.device
        models = session.entry.load_models(session._args("unused", "unused", "unused"))
        self.model, self.semantic_fn, self.f0_fn, _, self.campplus, self.to_mel, _ = models

    def analyse(self, wave44, with_style=False):
        import numpy as np
        import torchaudio
        torch = self.torch
        with torch.inference_mode():
            w44 = torch.from_numpy(wave44).float()[None].to(self.device)
            mel = self.to_mel(w44)
            w16 = torchaudio.functional.resample(w44, SR, 16000)
            content = self.semantic_fn(w16)
            f0 = self.f0_fn(w16[0], thred=0.03)
            f0 = np.asarray(f0, dtype=np.float64)
            style = None
            if with_style:
                feat = torchaudio.compliance.kaldi.fbank(w16, num_mel_bins=80, dither=0, sample_frequency=16000)
                feat = feat - feat.mean(dim=0, keepdim=True)
                style = self.campplus(feat.unsqueeze(0))[0].float().cpu()
        n50 = content.size(1)
        st, voiced = semitones(f0)
        idx = np.minimum(np.arange(n50) * 2, len(f0) - 1)
        return {"mel": mel[0], "content": content, "f0": f0, "w16": w16[0].cpu().numpy(),
                "st50": st[idx], "v50": voiced[idx], "e50": log_energy(w16[0].cpu().numpy(), n50),
                "style": style}

    def condition(self, content, f0, frames):
        torch = self.torch
        with torch.inference_mode():
            f0_t = torch.from_numpy(f0).float().to(self.device)[None]
            cond, *_ = self.model.length_regulator(content, ylens=torch.LongTensor([frames]).to(self.device),
                                                   n_quantizers=3, f0=f0_t)
        return cond[0]


def warp(extractor, src, tgt, mapping, shift_semitones):
    import numpy as np
    torch = extractor.torch
    content = src["content"][:, torch.from_numpy(mapping).to(src["content"].device)]
    n100 = len(tgt["f0"])
    k = np.arange(n100)
    idx = np.minimum(mapping[np.minimum(k // 2, len(mapping) - 1)] * 2 + (k % 2), len(src["f0"]) - 1)
    f0 = src["f0"][idx].copy()
    if shift_semitones:
        f0[f0 > 1] *= 2 ** (shift_semitones / 12)
    return extractor.condition(content, f0, tgt["mel"].size(-1))


def process_singer(extractor, singer_dir):
    singer = singer_dir.name
    straight, items, skipped = {}, [], []
    analysed = {}
    for set_name in SETS:
        base = singer_dir / set_name
        for vowel, files in takes(base / "straight").items():
            sid = f"{set_name}_{vowel}"
            wave = load_trimmed(files[0])
            info = extractor.analyse(wave, with_style=True)
            analysed[sid] = info
            cond = extractor.condition(info["content"], info["f0"], info["mel"].size(-1))
            straight[sid] = {"mel": info["mel"].half().cpu(), "cond": cond.half().cpu(), "style": info["style"],
                             "file": str(files[0].relative_to(ROOT)).replace("\\", "/"),
                             "seconds": len(wave) / SR}
    for set_name in SETS:
        base = singer_dir / set_name
        for folder, label in TECHNIQUES.items():
            for vowel, files in takes(base / folder).items():
                sid = f"{set_name}_{vowel}"
                if sid not in analysed:
                    skipped.append({"file": files[0].name, "reason": "no straight take"})
                    continue
                for path in files:
                    if "(" in path.name and len(files) > 1:
                        continue
                    rel = str(path.relative_to(ROOT)).replace("\\", "/")
                    try:
                        wave = load_trimmed(path)
                        tgt = extractor.analyse(wave)
                        src = analysed[sid]
                        mapping, diag = align(src, tgt, pitch_weight=0.3 if label == "vocal_fry" else 1.0)
                        shift = diag["key_offset"] if label != "vocal_fry" else 0
                        cond = warp(extractor, src, tgt, mapping, shift)
                        items.append({"tech": label, "file": rel, "straight": sid,
                                      "mel": tgt["mel"].half().cpu(), "cond": cond.half().cpu(),
                                      "seconds": len(wave) / SR, "shift_semitones": shift,
                                      "length_ratio": tgt["mel"].size(-1) / src["mel"].size(-1), **diag})
                    except Exception as error:
                        traceback.print_exc()
                        skipped.append({"file": rel, "reason": f"{type(error).__name__}: {error}"})
    return {"singer": singer, "straight": straight, "items": items, "skipped": skipped}


def main():
    import torch
    from probe_v0 import Session
    OUT.mkdir(parents=True, exist_ok=True)
    singers = sorted(p for p in VOCALSET.iterdir() if p.is_dir())
    if not singers:
        raise FileNotFoundError(f"no singers under {VOCALSET}")
    todo = [s for s in singers if not (OUT / f"{s.name}.pt").exists()]
    log(f"{len(singers)} singers, {len(todo)} to process")
    if todo:
        extractor = Extractor(Session())
        for singer_dir in todo:
            started = time.perf_counter()
            payload = process_singer(extractor, singer_dir)
            tmp = OUT / f"{singer_dir.name}.pt.tmp"
            torch.save(payload, tmp)
            tmp.replace(OUT / f"{singer_dir.name}.pt")
            log(f"{singer_dir.name}: {len(payload['straight'])} straight, {len(payload['items'])} pairs, "
                f"{len(payload['skipped'])} skipped, {time.perf_counter() - started:.0f}s")
    manifest = []
    for singer_dir in singers:
        payload = torch.load(OUT / f"{singer_dir.name}.pt", map_location="cpu")
        for item in payload["items"]:
            manifest.append({k: v for k, v in item.items() if k not in ("mel", "cond")} | {"singer": payload["singer"]})
        for skip in payload["skipped"]:
            manifest.append({"singer": payload["singer"], "skipped": True, **skip})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    pairs = [m for m in manifest if not m.get("skipped")]
    log(f"done: {len(pairs)} pairs, {sum(m['seconds'] for m in pairs) / 3600:.2f} h of targets, "
        f"{len(manifest) - len(pairs)} skipped")
    return 0


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    log(f"total wall {time.perf_counter() - started:.1f}s, exit {code}")
    raise SystemExit(code)
