"""Build the multi-listener test package.

Two trial types:
  I (identification): unmodified output vs. one controlled sample.
  E (matched EQ):     unmodified output, then the controlled sample and its matched
                      equaliser in random order (two calibration trials use the same audio twice).

Conditions (strengths fixed beforehand from the strength sweep of experiment 05, paper Section 6.4):
  adapter      fry 0.5/0.6, breathy 0.8/1.0, belt/forte/piano 1.0  (experiments 04/05)
  real         VocalSet recordings of held-out singers, plain vs. technique take
  source       Seed-VC output when the SOURCE itself is sung with the technique
  guidance     training-free multi-donor contrastive guidance, w = 4 (experiment 03)

Only the source condition needs the GPU (6 conversions). All audio is set to one common RMS
level and renamed to random codes. The page randomises trial order and A/B per listener.

Outputs:
  listening_test/                    folder that listeners open (START.html + audio/)
  多人听测包.zip                     the same folder zipped, to send to listeners
  results/listening_main/key.json    which code is which condition (researcher only, not in the zip)
"""
from __future__ import annotations

import json
import os
import random
import secrets
import shutil
import sys
import time
import traceback
import wave
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/steer"))
sys.path.insert(0, str(ROOT / "src"))

PKG = ROOT / "listening_test"
ZIP = ROOT / "多人听测包.zip"
RES = ROOT / "results/listening_main"
TEMPLATE = Path(__file__).resolve().parent / "listening_main_template.html"
TARGETS = {"f3": "data/style_pilot/identity_f3_6s.wav", "f9": "data/exp03_refs/target_f9.wav"}
CELLS = [("f3", "row"), ("f3", "dona"), ("f9", "row"), ("f9", "dona")]
TECH_DIR = {"breathy": "breathy", "belt": "belt", "vocal_fry": "vocal_fry", "forte": "slow_forte", "piano": "slow_piano"}
REAL_SINGERS = ["female3", "male9"]
SOURCE_SINGER, SOURCE_TARGET = "female1", "f9"
TF_AXES = {"breathy": "breathy", "belt": "belt", "vocal_fry": "fry", "forte": "forte"}
TARGET_DBFS = -26.0
PEAK_LIMIT = 0.97
CLIP_SECONDS = 6.0


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def adapter_wav(target, source, tech, amount):
    tag = f"{target}_{source}_{tech}_a{amount:g}.wav"
    folder = "results/exp04/eval/wav" if amount in (0.5, 1.0, 1.5) else "results/exp05/sweep/wav"
    return ROOT / folder / tag


def base_wav(target, source):
    return ROOT / f"results/exp04/eval/wav/{target}_{source}_base.wav"


def design():
    """Deterministic list of trials; every audio item is ('file', path) or ('make', recipe)."""
    trials = []

    def ident(cond, base, sample, **info):
        trials.append({"type": "I", "condition": cond, "base": base, "S": sample, **info})

    def eqt(cond, base, neural, eq, catch=False, **info):
        trials.append({"type": "E", "condition": cond, "base": base, "A": neural, "B": eq, "catch": catch, **info})

    for tech in ("belt", "forte", "piano"):
        for t, s in CELLS:
            ident("adapter", ("file", base_wav(t, s)), ("file", adapter_wav(t, s, tech, 1.0)),
                  technique=tech, strength=1.0, target=t, source=s)
    for tech, plan in (("vocal_fry", {0.5: (0, 3), 0.6: (1, 2)}), ("breathy", {0.8: (1, 2), 1.0: (0, 3)})):
        for amount, cells in plan.items():
            for i in cells:
                t, s = CELLS[i]
                ident("adapter", ("file", base_wav(t, s)), ("file", adapter_wav(t, s, tech, amount)),
                      technique=tech, strength=amount, target=t, source=s)
    for singer in REAL_SINGERS:
        for tech, folder in TECH_DIR.items():
            ident("real", ("make", {"kind": "clip", "singer": singer, "dir": "straight"}),
                  ("make", {"kind": "clip", "singer": singer, "dir": folder}), technique=tech, singer=singer)
    for tech, folder in TECH_DIR.items():
        ident("source", ("make", {"kind": "convert", "dir": "straight"}),
              ("make", {"kind": "convert", "dir": folder}), technique=tech, singer=SOURCE_SINGER, target=SOURCE_TARGET)
    for i, (tech, axis) in enumerate(TF_AXES.items()):
        t, s = CELLS[i % 4]
        ident("guidance", ("file", base_wav(t, s)), ("file", ROOT / f"results/exp03/wav/{t}_{s}_{axis}_cgm_w4.wav"),
              technique=tech, strength=4.0, target=t, source=s)
    for i, (tech, amount) in enumerate((("vocal_fry", 0.6), ("breathy", 1.0), ("belt", 1.0), ("forte", 1.0), ("piano", 1.0))):
        for j in (i % 4, (i + 2) % 4):
            t, s = CELLS[j]
            neural = adapter_wav(t, s, tech, amount)
            eq = ROOT / f"results/exp04/eval/wav/{t}_{s}_{tech}_eqmatch_a1.wav" if amount == 1.0 else \
                ("make", {"kind": "eqmatch", "neural": str(neural), "base": str(base_wav(t, s))})
            eqt("adapter", ("file", base_wav(t, s)), ("file", neural), eq if isinstance(eq, tuple) else ("file", eq),
                technique=tech, strength=amount, target=t, source=s)
    for tech, axis, j in (("breathy", "breathy", 3), ("vocal_fry", "fry", 1)):
        t, s = CELLS[j]
        eqt("guidance", ("file", base_wav(t, s)), ("file", ROOT / f"results/exp03/wav/{t}_{s}_{axis}_cgm_w4.wav"),
            ("file", ROOT / f"results/exp03/wav/{t}_{s}_{axis}_eqmatch_cgm_w4.wav"), technique=tech, strength=4.0, target=t, source=s)
    for tech, j in (("belt", 1), ("forte", 2)):
        t, s = CELLS[j]
        same = ("file", adapter_wav(t, s, tech, 1.0))
        eqt("catch", ("file", base_wav(t, s)), same, same, catch=True, technique=tech, strength=1.0, target=t, source=s)
    return trials


def read_wav(path):
    import librosa
    import numpy as np
    x, _ = librosa.load(str(path), sr=44100, mono=True)
    return np.asarray(x, dtype=np.float64)


def write_wav(path, x):
    import numpy as np
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
        w.writeframes(np.round(np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    import numpy as np
    from exp03_panel import find_take, cut_clip, ltas_db, third_octave_smooth, apply_eq, EQ_LIMIT_DB
    import exp03_panel

    exp03_panel.ROOT = ROOT
    RES.mkdir(parents=True, exist_ok=True)
    work = RES / "work"
    work.mkdir(exist_ok=True)
    trials = design()

    # ---- check inputs ----------------------------------------------------
    missing = sorted({str(item[1]) for t in trials for k in ("base", "S", "A", "B") if k in t
                      for item in [t[k]] if item[0] == "file" and not Path(item[1]).exists()})
    if missing:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(missing))

    # ---- materialise the 'make' items -------------------------------------
    made = {}
    session = None

    def make(recipe):
        nonlocal session
        key = json.dumps(recipe, sort_keys=True)
        if key in made:
            return made[key]
        kind = recipe["kind"]
        if kind == "clip":
            clip, info = cut_clip(find_take(recipe["singer"], recipe["dir"]), seconds=CLIP_SECONDS)
            path = work / f"real_{recipe['singer']}_{recipe['dir']}.wav"
            write_wav(path, clip)
        elif kind == "convert":
            src = work / f"source_{SOURCE_SINGER}_{recipe['dir']}.wav"
            clip, _ = cut_clip(find_take(SOURCE_SINGER, recipe["dir"]), seconds=CLIP_SECONDS)
            write_wav(src, clip)
            if session is None:
                from probe_v0 import Session
                session = Session()
                session.tap.close()
            path = work / f"converted_{SOURCE_SINGER}_{recipe['dir']}_to_{SOURCE_TARGET}.wav"
            session.convert(str(src.relative_to(ROOT)), TARGETS[SOURCE_TARGET], path)
        elif kind == "eqmatch":
            neural, base = read_wav(recipe["neural"]), read_wav(recipe["base"])
            freqs = np.fft.rfftfreq(2048, 1.0 / 44100)
            gain = third_octave_smooth(freqs, ltas_db(neural) - ltas_db(base))
            gain[freqs < 60] = 0.0
            path = work / ("eqmatch_" + Path(recipe["neural"]).name)
            write_wav(path, apply_eq(base.astype(np.float32), np.clip(gain, -EQ_LIMIT_DB, EQ_LIMIT_DB)))
        else:
            raise ValueError(kind)
        made[key] = ("file", path)
        log(f"made {Path(path).name}")
        return made[key]

    for t in trials:
        for k in ("base", "S", "A", "B"):
            if k in t and t[k][0] == "make":
                t[k] = make(t[k][1])

    # ---- one common level, coded names -------------------------------------
    rng = random.Random(secrets.randbits(64))
    sources = sorted({str(t[k][1]) for t in trials for k in ("base", "S", "A", "B") if k in t})
    audio = {s: read_wav(s) for s in sources}
    gains = {s: 10 ** (TARGET_DBFS / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-12) for s, x in audio.items()}
    worst = max(np.max(np.abs(x)) * gains[s] for s, x in audio.items())
    trim = min(1.0, PEAK_LIMIT / worst)
    level = TARGET_DBFS + 20 * np.log10(trim)
    log(f"{len(sources)} distinct audio files; common level {level:.1f} dBFS RMS (peak limit {PEAK_LIMIT})")

    if PKG.exists():
        shutil.rmtree(PKG)
    (PKG / "audio").mkdir(parents=True)
    codes = rng.sample(range(10000, 100000), 4 * len(trials) + len(sources))
    code_iter = iter(codes)
    coded = {}

    def code_for(src, fresh=False):
        if src in coded and not fresh:
            return coded[src]
        name = f"{next(code_iter)}.wav"
        write_wav(PKG / "audio" / name, audio[src] * gains[src] * trim)
        if not fresh:
            coded[src] = name
        return name

    order = list(range(len(trials)))
    rng.shuffle(order)  # trial ids are assigned in random order so ids carry no information
    manifest, key = [], []
    for new_id, idx in enumerate(order, 1):
        t = trials[idx]
        tid = f"{t['type']}{new_id:02d}"
        files = {k: code_for(str(t[k][1]), fresh=(t.get("catch") and k == "B")) for k in ("base", "S", "A", "B") if k in t}
        manifest.append({"id": tid, "type": t["type"], **{k: "audio/" + v for k, v in files.items()}})
        key.append({"id": tid, **{k: v for k, v in t.items() if k not in ("base", "S", "A", "B")},
                    "files": files, "sources": {k: os.path.relpath(str(t[k][1]), ROOT).replace("\\", "/") for k in files}})
    calib = next(m for m in manifest if m["type"] == "I")  # a volume-setting sample: any plain base
    volume_file = calib["base"]

    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__MANIFEST__", json.dumps(manifest)).replace("__VOLUME__", json.dumps(volume_file))
    (PKG / "START.html").write_text(html, encoding="utf-8")
    (RES / "key.json").write_text(json.dumps({"level_dbfs": level, "trials": key}, ensure_ascii=False, indent=1), encoding="utf-8")

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(PKG.rglob("*")):
            if path.is_file():
                z.write(path, Path("listening_test") / path.relative_to(PKG))
    n_i = sum(m["type"] == "I" for m in manifest)
    log(f"done: {len(manifest)} trials ({n_i} identification, {len(manifest) - n_i} matched-EQ), "
        f"{len(list((PKG / 'audio').iterdir()))} audio files, zip {ZIP.stat().st_size / 2 ** 20:.1f} MB")
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
