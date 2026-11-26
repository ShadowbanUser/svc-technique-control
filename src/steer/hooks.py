"""Record and steer the residual stream of a frozen Seed-VC flow-matching DiT.

Nothing here trains or modifies weights. A CfmTap wraps one `model.cfm` object:

* it tracks which Euler step the solver is on and how many leading frames
  belong to the reference prompt;
* in record mode it stores, for every (step, layer), the mean activation over
  the generated frames of the conditional branch;
* in steer mode it adds `alpha * vector[layer]` to the generated frames.

The DiT is called once per Euler step. With classifier-free guidance the batch
is [conditional, unconditional], so row 0 is the conditional branch.
"""
from __future__ import annotations

import torch


class CfmTap:
    def __init__(self, cfm):
        self.cfm = cfm
        est = cfm.estimator
        self.layers = list(est.transformer.layers)
        self.n_layers = len(self.layers)
        # Extra tokens prepended to the sequence shift frame indices.
        self.offset = int(bool(getattr(est, "time_as_token", False))) + \
            int(bool(getattr(est, "style_as_token", False)))
        self.step = -1
        self.prompt_len = 0
        self.solver_calls = 0
        self._record = None
        self._steer = None
        self._style_delta = None   # added to the style (speaker) vector
        self._guide = None         # contrastive guidance settings
        self._capture = None       # filled with the solver inputs of the next call
        self._aux_pass = False     # True while running an extra pass that hooks must ignore
        self._manual_steps = False
        self._orig_solve = cfm.solve_euler
        cfm.solve_euler = self._solve  # instance attribute shadows the bound method
        self._handles = [est.register_forward_pre_hook(self._pre)]
        for index, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(index)))

    # ---- lifecycle -----------------------------------------------------
    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if "solve_euler" in self.cfm.__dict__:
            del self.cfm.__dict__["solve_euler"]

    def _solve(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
        self.prompt_len = int(prompt.size(-1))
        self.step = -1
        self.solver_calls += 1
        if self._capture is not None:
            self._capture.append({"prompt": prompt.detach().clone(), "mu": mu.detach().clone(),
                                  "style": style.detach().clone()})
        if self._style_delta is not None:
            style = style + self._style_delta.to(device=style.device, dtype=style.dtype)
        if self._guide is not None:
            return self._solve_guided(x, x_lens, prompt, mu, style, t_span, inference_cfg_rate)
        return self._orig_solve(x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def _pre(self, module, args):
        if not self._manual_steps:
            self.step += 1

    # ---- conditioning capture / style shift / contrastive guidance -------
    def start_capture(self):
        self._capture = []

    def stop_capture(self):
        """Return the solver inputs (reference mel, content sequence, style vector) of one call."""
        captured, self._capture = self._capture, None
        if not captured or len(captured) != 1:
            raise RuntimeError(f"Expected exactly one solver call, got {0 if not captured else len(captured)}")
        return captured[0]

    def set_style_delta(self, delta=None):
        self._style_delta = delta

    def set_guidance(self, positive=None, negative=None, weight=0.0):
        """Contrastive guidance: velocity += weight * (v(positive) - v(negative)).

        positive / negative are packs from stop_capture() made with the SAME source audio,
        e.g. a donor singer's breathy and plain references. Disabled when weight == 0.
        """
        if positive is None or negative is None or weight == 0:
            self._guide = None
            return
        positive = list(positive) if isinstance(positive, (list, tuple)) else [positive]
        negative = list(negative) if isinstance(negative, (list, tuple)) else [negative]
        if len(positive) != len(negative) or not positive:
            raise ValueError("positive and negative must be non-empty and of equal length")
        self._guide = {"positive": positive, "negative": negative, "weight": float(weight)}

    def _contrast(self, x_gen, t):
        """Mean over donor pairs of v(positive) - v(negative) on the generated frames."""
        guide = self._guide
        positive, negative = guide["positive"], guide["negative"]
        packs = positive + negative
        same_length = len({int(pack["prompt"].size(-1)) for pack in packs}) == 1
        if same_length and len(packs) > 2:
            est = self.cfm.estimator
            count = len(packs)
            prompt = torch.cat([pack["prompt"] for pack in packs], dim=0).to(device=x_gen.device, dtype=x_gen.dtype)
            prompt_len, frames = prompt.size(-1), x_gen.size(-1)
            mu = torch.cat([pack["mu"] for pack in packs], dim=0).to(device=x_gen.device)
            if mu.size(1) != prompt_len + frames:
                raise RuntimeError(f"Guidance packs have {mu.size(1) - prompt_len} generated frames, need {frames}")
            style = torch.cat([pack["style"] for pack in packs], dim=0).to(device=x_gen.device)
            x_rep = x_gen.expand(count, -1, -1)
            x_full = torch.cat([torch.zeros_like(prompt), x_rep], dim=-1)
            prompt_x = torch.cat([prompt, torch.zeros_like(x_rep)], dim=-1)
            x_lens = torch.LongTensor([prompt_len + frames]).to(x_gen.device)
            self._aux_pass = True
            try:
                out = est(x_full, prompt_x, x_lens, t.unsqueeze(0).expand(count), style, mu)[..., prompt_len:]
            finally:
                self._aux_pass = False
            half = len(positive)
            return out[:half].float().mean(0, keepdim=True) - out[half:].float().mean(0, keepdim=True)
        total = 0.0
        for pos, neg in zip(positive, negative):
            total = total + (self._branch_velocity(pos, x_gen, t).float() - self._branch_velocity(neg, x_gen, t).float())
        return total / len(positive)

    def _branch_velocity(self, pack, x_gen, t):
        """Conditional velocity of the generated frames under another reference."""
        est = self.cfm.estimator
        prompt = pack["prompt"].to(device=x_gen.device, dtype=x_gen.dtype)
        prompt_len = prompt.size(-1)
        frames = x_gen.size(-1)
        mu = pack["mu"].to(device=x_gen.device)
        if mu.size(1) != prompt_len + frames:
            raise RuntimeError(f"Guidance pack has {mu.size(1) - prompt_len} generated frames, need {frames}")
        x_full = torch.cat([torch.zeros_like(prompt), x_gen], dim=-1)
        prompt_x = torch.cat([prompt, torch.zeros_like(x_gen)], dim=-1)
        x_lens = torch.LongTensor([prompt_len + frames]).to(x_gen.device)
        style = pack["style"].to(device=x_gen.device)
        self._aux_pass = True
        try:
            out = est(x_full, prompt_x, x_lens, t.unsqueeze(0), style, mu)
        finally:
            self._aux_pass = False
        return out[..., prompt_len:]

    def _solve_guided(self, x, x_lens, prompt, mu, style, t_span, inference_cfg_rate):
        """Upstream fixed-step Euler solver plus the contrastive term on generated frames."""
        cfm, guide = self.cfm, self._guide
        t = t_span[0]
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if cfm.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0
        self._manual_steps = True
        try:
            for step in range(1, len(t_span)):
                self.step = step - 1
                dt = t_span[step] - t_span[step - 1]
                if inference_cfg_rate > 0:
                    stacked = cfm.estimator(
                        torch.cat([x, x], dim=0),
                        torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0),
                        x_lens,
                        torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0),
                        torch.cat([style, torch.zeros_like(style)], dim=0),
                        torch.cat([mu, torch.zeros_like(mu)], dim=0))
                    dphi_dt, cfg_dphi_dt = stacked.chunk(2, dim=0)
                    dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
                else:
                    dphi_dt = cfm.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)
                x_gen = x[..., prompt_len:]
                contrast = self._contrast(x_gen, t).to(dphi_dt.dtype)
                dphi_dt = dphi_dt.clone()
                dphi_dt[..., prompt_len:] = dphi_dt[..., prompt_len:] + guide["weight"] * contrast
                x = x + dt * dphi_dt
                t = t + dt
                x[:, :, :prompt_len] = 0
        finally:
            self._manual_steps = False
        return x

    # ---- modes ---------------------------------------------------------
    def start_recording(self):
        self._record = {"sum": {}, "norm": {}, "count": {}}

    def stop_recording(self):
        """Return {'mean': [steps, layers, dim], 'norm': [steps, layers]} on CPU."""
        rec, self._record = self._record, None
        if not rec or not rec["sum"]:
            raise RuntimeError("Nothing was recorded")
        steps = 1 + max(step for step, _ in rec["sum"])
        dim = next(iter(rec["sum"].values())).numel()
        mean = torch.zeros(steps, self.n_layers, dim, dtype=torch.float64)
        norm = torch.zeros(steps, self.n_layers, dtype=torch.float64)
        for (step, layer), total in rec["sum"].items():
            count = rec["count"][step, layer]
            mean[step, layer] = total / count
            norm[step, layer] = rec["norm"][step, layer] / count
        return {"mean": mean.float(), "norm": norm.float()}

    def set_steering(self, vectors=None, alpha=0.0, branch="cond"):
        """vectors: {layer_index: tensor [dim] or [steps, dim]}; None disables."""
        if vectors is None or alpha == 0:
            self._steer = None
            return
        if branch not in ("cond", "both"):
            raise ValueError("branch must be 'cond' or 'both'")
        self._steer = {"vectors": vectors, "alpha": float(alpha), "branch": branch}

    # ---- hook ----------------------------------------------------------
    def _make_hook(self, index):
        def hook(module, args, output):
            if self._aux_pass:
                return None
            start = self.offset + self.prompt_len
            steer = self._steer
            changed = False
            if steer is not None and index in steer["vectors"]:
                vector = steer["vectors"][index]
                if vector.dim() == 2:
                    vector = vector[min(self.step, vector.size(0) - 1)]
                delta = (steer["alpha"] * vector).to(device=output.device, dtype=output.dtype)
                output = output.clone()
                if steer["branch"] == "cond":
                    output[:1, start:, :] += delta
                else:
                    output[:, start:, :] += delta
                changed = True
            rec = self._record
            if rec is not None:
                frames = output[0, start:, :].detach().float()
                key = (self.step, index)
                if frames.size(0) > 0:
                    mean = frames.mean(0).double().cpu()
                    norm = float(frames.norm(dim=-1).mean())
                    rec["sum"][key] = rec["sum"].get(key, 0) + mean
                    rec["norm"][key] = rec["norm"].get(key, 0.0) + norm
                    rec["count"][key] = rec["count"].get(key, 0) + 1
            return output if changed else None
        return hook


def difference_of_means(positive, negative):
    """positive/negative: lists of recordings from stop_recording().

    Returns {'direction': [steps, layers, dim], 'relative_norm': [steps, layers]}
    where relative_norm is |direction| divided by the mean activation norm.
    """
    pos = torch.stack([item["mean"] for item in positive]).mean(0)
    neg = torch.stack([item["mean"] for item in negative]).mean(0)
    scale = torch.stack([item["norm"] for item in positive + negative]).mean(0)
    direction = pos - neg
    return {"direction": direction,
            "relative_norm": direction.norm(dim=-1) / scale.clamp_min(1e-8)}


def layer_vectors(direction, layers, per_step=False):
    """Pick layers out of a [steps, layers, dim] direction.

    per_step=False averages over Euler steps and gives one vector per layer.
    """
    out = {}
    for layer in layers:
        out[int(layer)] = direction[:, layer, :].clone() if per_step \
            else direction[:, layer, :].mean(0)
    return out
