"""Train technique-gated adapters on frozen Seed-VC (flow-matching loss).

Inputs come from cache/techpairs (prep_technique_pairs.py). Only the adapters
are trained; every Seed-VC weight stays frozen. Five singers are held out and
used only for validation:

  val loss with the true technique label vs. with the adapter off (= Seed-VC)
  -> does the label carry technique information that generalises to unseen singers?
  confusion matrix: loss for every (true technique, label used) pair
  -> is each label specific, or do all labels do the same thing?

Outputs: results/exp04/train/ (adapter_best.pt, adapter_final.pt, log.jsonl, summary.json)
"""
from __future__ import annotations

import contextlib
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/steer"))
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results/exp04/train"
HELD_OUT = ("female1", "female3", "female9", "male3", "male9")
CONFIG = {
    "rank": 40, "steps": 4000, "batch": 8, "lr": 2e-4, "warmup": 200, "min_lr_ratio": 0.1,
    "grad_clip": 1.0, "null_prob": 0.1, "sigma_min": 1e-6,
    "target_frames": [215, 430], "prompt_frames": [172, 344],
    "drop_worst_alignment": 0.1, "val_every": 500, "val_per_tech": 24, "seed": 20261117,
}
FPS = 44100 / 512


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def drop_set(cache_dir, techniques):
    """Files whose alignment cost is in the worst DROP fraction of their technique (from manifest.json)."""
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    pairs = [m for m in manifest if not m.get("skipped") and m["tech"] in techniques]
    dropped = []
    for tech in techniques:
        group = sorted((m for m in pairs if m["tech"] == tech), key=lambda m: m["path_cost"])
        cut = int(len(group) * (1 - CONFIG["drop_worst_alignment"]))
        dropped += [(m["file"], round(m["path_cost"], 3)) for m in group[cut:]]
    return dropped


def load_cache(techniques, cache_dir, keep=None, drop_files=frozenset()):
    import torch
    data = {}
    for path in sorted(cache_dir.glob("*.pt")):
        if keep is not None and path.stem not in keep:
            continue
        payload = torch.load(path, map_location="cpu")
        data[payload["singer"]] = payload
    if not data:
        raise FileNotFoundError(f"no cache in {cache_dir}; run the preparation step first")
    items = [dict(item, singer=s) for s, p in data.items() for item in p["items"]
             if item["tech"] in techniques and item["file"] not in drop_files]
    return data, items


class Batcher:
    def __init__(self, data, items, techniques, rng):
        self.data, self.techniques, self.rng = data, techniques, rng
        self.by_tech = {t: [i for i in items if i["tech"] == t] for t in techniques}
        self.by_tech = {t: v for t, v in self.by_tech.items() if v}

    def example(self, item, rng):
        import torch
        refs = self.data[item["singer"]]["straight"]
        choices = [k for k in refs if k != item["straight"]] or list(refs)
        ref = refs[rng.choice(sorted(choices))]
        lo, hi = CONFIG["target_frames"]
        t_len = min(rng.randint(lo, hi), item["mel"].size(-1))
        t0 = rng.randint(0, item["mel"].size(-1) - t_len)
        lo, hi = CONFIG["prompt_frames"]
        p_len = min(rng.randint(lo, hi), ref["mel"].size(-1))
        p0 = rng.randint(0, ref["mel"].size(-1) - p_len)
        mel = torch.cat([ref["mel"][:, p0:p0 + p_len], item["mel"][:, t0:t0 + t_len]], dim=-1).float()
        cond = torch.cat([ref["cond"][p0:p0 + p_len], item["cond"][t0:t0 + t_len]], dim=0).float()
        alpha = torch.zeros(len(self.techniques))
        alpha[self.techniques.index(item["tech"])] = 1.0
        return {"mel": mel, "cond": cond, "style": ref["style"].float(), "prompt": p_len, "alpha": alpha}

    def sample(self, n):
        out = []
        for _ in range(n):
            tech = self.rng.choice(sorted(self.by_tech))
            out.append(self.example(self.rng.choice(self.by_tech[tech]), self.rng))
        return out


def collate(examples, device):
    import torch
    lengths = [e["mel"].size(-1) for e in examples]
    total = max(lengths)
    b = len(examples)
    mel = torch.zeros(b, 128, total)
    cond = torch.zeros(b, total, examples[0]["cond"].size(-1))
    for i, e in enumerate(examples):
        mel[i, :, :lengths[i]] = e["mel"]
        cond[i, :lengths[i]] = e["cond"]
    return {"mel": mel.to(device), "cond": cond.to(device),
            "style": torch.stack([e["style"] for e in examples]).to(device),
            "prompt": torch.LongTensor([e["prompt"] for e in examples]).to(device),
            "lens": torch.LongTensor(lengths).to(device),
            "alpha": torch.stack([e["alpha"] for e in examples]).to(device)}


def flow_loss(estimator, adapter, batch, t=None, noise=None, null_rows=None, alpha=None, per_row=False):
    """Seed-VC's conditional flow-matching L1 loss on the generated (non-prompt) frames."""
    import torch
    x1 = batch["mel"]
    b = x1.size(0)
    sigma = CONFIG["sigma_min"]
    t = torch.rand(b, device=x1.device) if t is None else t
    z = torch.randn_like(x1) if noise is None else noise
    tt = t.view(b, 1, 1)
    y = (1 - (1 - sigma) * tt) * z + tt * x1
    u = x1 - (1 - sigma) * z
    frames = torch.arange(x1.size(-1), device=x1.device)[None, :]
    in_prompt = (frames < batch["prompt"][:, None])[:, None, :]
    prompt_x = torch.where(in_prompt, x1, torch.zeros_like(x1))
    y = torch.where(in_prompt, torch.zeros_like(y), y)
    style, cond = batch["style"], batch["cond"]
    if null_rows is not None and bool(null_rows.any()):
        keep = (~null_rows).float()
        prompt_x = prompt_x * keep.view(b, 1, 1)
        style = style * keep.view(b, 1)
        cond = cond * keep.view(b, 1, 1)
    adapter.set(batch["alpha"] if alpha is None else alpha)
    out = estimator(y, prompt_x, batch["lens"], t, style, cond)
    valid = (frames >= batch["prompt"][:, None]) & (frames < batch["lens"][:, None])
    err = (out.float() - u.float()).abs().mean(dim=1)
    row = (err * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    return row if per_row else row.mean()


def build_val(batcher_items, data, techniques, device):
    """A fixed validation set: crops, noise and timesteps are all seeded."""
    import torch
    rng = random.Random(CONFIG["seed"] + 1)
    batcher = Batcher(data, batcher_items, techniques, rng)
    sets = {}
    gen = torch.Generator().manual_seed(CONFIG["seed"] + 2)
    for tech, pool in batcher.by_tech.items():
        chosen = [rng.choice(pool) for _ in range(CONFIG["val_per_tech"])]
        examples = [batcher.example(item, rng) for item in chosen]
        batches = []
        for i in range(0, len(examples), 4):
            batch = collate(examples[i:i + 4], "cpu")
            n = batch["mel"].size(0)
            batch["t"] = torch.linspace(0.15, 0.85, n)[torch.randperm(n, generator=gen)]
            batch["noise"] = torch.randn(batch["mel"].shape, generator=gen)
            batches.append({k: v.to(device) for k, v in batch.items()})
        sets[tech] = batches
    return sets


def evaluate(estimator, adapter, val_sets, techniques, autocast, full=False):
    import torch
    results = {}
    with torch.no_grad(), autocast():
        for tech, batches in val_sets.items():
            def run(alpha_fn):
                total, count = 0.0, 0
                for batch in batches:
                    rows = flow_loss(estimator, adapter, batch, t=batch["t"], noise=batch["noise"],
                                     alpha=alpha_fn(batch), per_row=True)
                    total += float(rows.sum()); count += rows.numel()
                return total / count
            entry = {"true_label": run(lambda b: b["alpha"]),
                     "adapter_off": run(lambda b: torch.zeros_like(b["alpha"]))}
            if full:
                for k, other in enumerate(techniques):
                    def one_hot(b, k=k):
                        a = torch.zeros_like(b["alpha"]); a[:, k] = 1.0
                        return a
                    entry[f"label_{other}"] = run(one_hot)
            results[tech] = entry
    adapter.off()
    return results


def main(argv=None):
    """Default: train and validate on cache/techpairs (experiment 04).
    --train-cache lets an ablation train on a different cache while validating on the
    standard one, with exactly the same training/validation pairs."""
    import argparse
    import torch
    from probe_v0 import Session
    from steer.adapter import TechniqueAdapter, DEFAULT_TECHNIQUES
    global OUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", default="cache/techpairs")
    parser.add_argument("--val-cache", default="cache/techpairs")
    parser.add_argument("--out", default="results/exp04/train")
    args = parser.parse_args(argv)
    train_dir, val_dir, OUT = (ROOT / args.train_cache, ROOT / args.val_cache, ROOT / args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "summary.json").exists():
        log(f"{OUT} already has summary.json; nothing to do")
        return 0
    techniques = list(DEFAULT_TECHNIQUES)
    rng = random.Random(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    dropped = drop_set(val_dir, techniques)
    drop_files = frozenset(f for f, _ in dropped)
    singers = {p.stem for p in val_dir.glob("*.pt")}
    train_data, train_items = load_cache(techniques, train_dir, singers - set(HELD_OUT), drop_files)
    val_data, val_items = load_cache(techniques, val_dir, set(HELD_OUT), drop_files)
    data = {**train_data, **val_data}
    log(f"training cache {args.train_cache}, validation cache {args.val_cache}, output {args.out}")
    log(f"{len(train_items)} training pairs, {len(val_items)} held-out pairs, {len(dropped)} dropped for poor alignment")
    for tech in techniques:
        log(f"  {tech}: {sum(i['tech'] == tech for i in train_items)} train / {sum(i['tech'] == tech for i in val_items)} val")

    session = Session()
    device = session.entry.device
    session.tap.close()  # the analysis hooks from earlier experiments are not needed here
    estimator = session.model.cfm.estimator
    estimator.eval()
    for p in session.model.cfm.parameters():
        p.requires_grad_(False)
    estimator.setup_caches(max_batch_size=CONFIG["batch"], max_seq_length=8192)
    adapter = TechniqueAdapter(estimator, techniques, CONFIG["rank"])
    params = list(adapter.parameters())
    for p in params:
        p.requires_grad_(True)
    log(f"adapter: {adapter.n_parameters() / 1e6:.2f} M parameters in {len(adapter.wrapped)} layers "
        f"(Seed-VC DiT itself: {sum(p.numel() for p in estimator.parameters()) / 1e6 - adapter.n_parameters() / 1e6:.1f} M, frozen)")

    on_cuda = str(device).startswith("cuda")
    bf16 = on_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if on_cuda else torch.float32)
    scaler = torch.amp.GradScaler("cuda", enabled=on_cuda and not bf16)
    if on_cuda:
        autocast = lambda: torch.autocast(device_type="cuda", dtype=dtype)  # noqa: E731
    else:
        autocast = contextlib.nullcontext
    opt = torch.optim.AdamW(params, lr=CONFIG["lr"], weight_decay=0.0)

    def lr_at(step):
        if step < CONFIG["warmup"]:
            return CONFIG["lr"] * (step + 1) / CONFIG["warmup"]
        progress = (step - CONFIG["warmup"]) / max(1, CONFIG["steps"] - CONFIG["warmup"])
        floor = CONFIG["min_lr_ratio"]
        return CONFIG["lr"] * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))

    batcher = Batcher(data, train_items, techniques, rng)
    val_sets = build_val(val_items, data, techniques, device)
    log_path = OUT / "log.jsonl"
    log_path.write_text("", encoding="utf-8")
    before = evaluate(estimator, adapter, val_sets, techniques, autocast, full=False)
    log("val before training (adapter has no effect yet): " +
        ", ".join(f"{t} {v['true_label']:.4f}" for t, v in before.items()))

    micro = CONFIG["batch"]
    best, best_step, ema, started = float("inf"), -1, None, time.perf_counter()
    step = 0
    while step < CONFIG["steps"]:
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        examples = batcher.sample(CONFIG["batch"])
        opt.zero_grad(set_to_none=True)
        try:
            total = 0.0
            for i in range(0, len(examples), micro):
                chunk = examples[i:i + micro]
                batch = collate(chunk, device)
                null = torch.rand(len(chunk), device=device) < CONFIG["null_prob"]
                with autocast():
                    loss = flow_loss(estimator, adapter, batch, null_rows=null) * len(chunk) / len(examples)
                scaler.scale(loss).backward()
                total += float(loss.detach())
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            adapter.off()
            torch.cuda.empty_cache()
            if micro == 1:
                raise
            micro = max(1, micro // 2)
            log(f"out of GPU memory; retrying with micro-batches of {micro}")
            continue
        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, CONFIG["grad_clip"]))
        scaler.step(opt)
        scaler.update()
        adapter.off()
        step += 1
        ema = total if ema is None else 0.98 * ema + 0.02 * total
        if step % 50 == 0 or step == 1:
            rate = step / (time.perf_counter() - started)
            log(f"step {step}/{CONFIG['steps']} loss {ema:.4f} grad {grad_norm:.3f} lr {lr_at(step):.2e} "
                f"{rate:.2f} it/s, eta {(CONFIG['steps'] - step) / rate / 60:.0f} min, "
                f"gpu peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
        if step % CONFIG["val_every"] == 0 or step == CONFIG["steps"]:
            val = evaluate(estimator, adapter, val_sets, techniques, autocast, full=False)
            score = sum(v["true_label"] for v in val.values()) / len(val)
            gain = {t: round(v["adapter_off"] - v["true_label"], 5) for t, v in val.items()}
            log(f"val step {step}: mean {score:.4f}; gain over Seed-VC per technique {gain}")
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, "train_loss": ema, "val": val}) + "\n")
            if score < best:
                best, best_step = score, step
                torch.save(adapter.state_dict(), OUT / "adapter_best.pt")
        if step % 1000 == 0:
            torch.save(adapter.state_dict(), OUT / f"adapter_step{step}.pt")
    torch.save(adapter.state_dict(), OUT / "adapter_final.pt")

    adapter.load_state_dict(torch.load(OUT / "adapter_best.pt", map_location="cpu"))
    confusion = evaluate(estimator, adapter, val_sets, techniques, autocast, full=True)
    summary = {"status": "completed", "config": CONFIG, "train_cache": args.train_cache, "val_cache": args.val_cache, "techniques": techniques, "held_out": HELD_OUT,
               "train_pairs": len(train_items), "val_pairs": len(val_items), "dropped": dropped,
               "best_step": best_step, "best_val": best, "confusion_best": confusion,
               "minutes": (time.perf_counter() - started) / 60,
               "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30, "dtype": str(dtype)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"done: best step {best_step}, val {best:.4f}")
    for tech, row in confusion.items():
        log(f"  true {tech:9s} " + " ".join(f"{k.replace('label_', '')}={v:.4f}" for k, v in row.items()))
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
