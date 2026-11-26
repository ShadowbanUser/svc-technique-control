"""Experiment 05, part 1: two checks on the experiment-04 adapter.

(a) Domain control. On held-out singers' STRAIGHT takes (no technique at all), compare the
    flow-matching loss with the adapter off and with each technique label. If labels lower
    the loss here too, part of the adapter's gain is adaptation to VocalSet recordings,
    not technique.
(b) Strength sweep. The exp04 test cells at technique strengths 0.6-0.9, to find the range that
    is audible but not distorted.

Outputs: results/exp05/domain_control.json, results/exp05/sweep/
"""
from __future__ import annotations

import contextlib
import json
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/train"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp04_adapter_eval as ev  # noqa: E402
import train_technique_adapter as tr  # noqa: E402
from probe_v0 import log  # noqa: E402

OUT = ROOT / "results/exp05"
CACHE = ROOT / "cache/techpairs"
SWEEP = [0.6, 0.7, 0.8, 0.9]
REPEATS = 2


def domain_control():
    import torch
    from probe_v0 import Session
    from steer.adapter import TechniqueAdapter

    session = Session()
    session.tap.close()
    estimator = session.model.cfm.estimator
    adapter = TechniqueAdapter.from_file(estimator, ROOT / "results/exp04/train/adapter_best.pt")
    techniques = adapter.techniques
    device = session.entry.device
    on_cuda = str(device).startswith("cuda")
    dtype = torch.bfloat16 if on_cuda and torch.cuda.is_bf16_supported() else torch.float16
    autocast = (lambda: torch.autocast(device_type="cuda", dtype=dtype)) if on_cuda else contextlib.nullcontext

    data = {s: torch.load(CACHE / f"{s}.pt", map_location="cpu") for s in tr.HELD_OUT if (CACHE / f"{s}.pt").exists()}
    if not data:
        raise FileNotFoundError(f"no held-out singer caches in {CACHE}")
    rng = random.Random(tr.CONFIG["seed"] + 5)
    batcher = tr.Batcher(data, [], techniques, rng)
    examples = []
    for singer, payload in data.items():
        for sid, entry in sorted(payload["straight"].items()):
            item = {"mel": entry["mel"], "cond": entry["cond"], "singer": singer, "straight": sid, "tech": techniques[0]}
            examples += [batcher.example(item, rng) for _ in range(REPEATS)]
    gen = torch.Generator().manual_seed(tr.CONFIG["seed"] + 6)
    batches = []
    for i in range(0, len(examples), 4):
        batch = tr.collate(examples[i:i + 4], "cpu")
        n = batch["mel"].size(0)
        batch["t"] = torch.linspace(0.15, 0.85, n)[torch.randperm(n, generator=gen)]
        batch["noise"] = torch.randn(batch["mel"].shape, generator=gen)
        batches.append({k: v.to(device) for k, v in batch.items()})

    def run(alpha_row):
        total, count = 0.0, 0
        with torch.no_grad(), autocast():
            for batch in batches:
                alpha = alpha_row.to(device).expand(batch["mel"].size(0), -1)
                rows = tr.flow_loss(estimator, adapter, batch, t=batch["t"], noise=batch["noise"], alpha=alpha, per_row=True)
                total += float(rows.sum()); count += rows.numel()
        adapter.off()
        return total / count

    result = {"n_examples": len(examples), "adapter_off": run(torch.zeros(1, len(techniques)))}
    for k, tech in enumerate(techniques):
        alpha = torch.zeros(1, len(techniques)); alpha[0, k] = 1.0
        result[f"label_{tech}"] = run(alpha)
    exp04 = json.loads((ROOT / "results/exp04/train/summary.json").read_text(encoding="utf-8"))["confusion_best"]
    result["technique_data_for_reference"] = exp04
    log("straight takes of held-out singers: " + ", ".join(f"{k}={v:.4f}" for k, v in result.items() if isinstance(v, float)))
    adapter.remove()
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "domain_control.json"
    if not path.exists():
        path.write_text(json.dumps(domain_control(), indent=2), encoding="utf-8")
    ev.OUT = OUT / "sweep"
    ev.AMOUNTS = SWEEP
    ev.COMBOS = {}
    return ev.main()


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    log(f"total wall {time.perf_counter() - started:.1f}s, exit {code}")
    raise SystemExit(code)
