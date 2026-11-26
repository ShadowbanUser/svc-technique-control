"""Ablation cache: the same pairs as cache/techpairs, but the content and F0 inputs come
from the technique take itself (no DTW, no straight take). Everything else is identical.

Output: cache/techpairs_self/<singer>.pt and a copy of manifest.json
"""
from __future__ import annotations

import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/steer"))
sys.path.insert(0, str(ROOT / "scripts/train"))
sys.path.insert(0, str(ROOT / "src"))

from prep_technique_pairs import Extractor, load_trimmed, log  # noqa: E402

SRC = ROOT / "cache/techpairs"
OUT = ROOT / "cache/techpairs_self"


def main():
    import torch
    from probe_v0 import Session
    OUT.mkdir(parents=True, exist_ok=True)
    singers = sorted(SRC.glob("*.pt"))
    if not singers:
        raise FileNotFoundError(f"no cache in {SRC}")
    todo = [p for p in singers if not (OUT / p.name).exists()]
    log(f"{len(singers)} singers, {len(todo)} to process")
    extractor = Extractor(Session()) if todo else None
    for path in todo:
        started = time.perf_counter()
        payload = torch.load(path, map_location="cpu")
        for item in payload["items"]:
            info = extractor.analyse(load_trimmed(ROOT / item["file"]))
            frames = item["mel"].size(-1)
            if info["mel"].size(-1) != frames:
                raise RuntimeError(f"{item['file']}: mel length {info['mel'].size(-1)} != cached {frames}")
            item["cond"] = extractor.condition(info["content"], info["f0"], frames).half().cpu()
            item["content_source"] = "self"
        tmp = OUT / (path.name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(OUT / path.name)
        log(f"{path.stem}: {len(payload['items'])} items, {time.perf_counter() - started:.0f}s")
    shutil.copyfile(SRC / "manifest.json", OUT / "manifest.json")
    log("done")
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
