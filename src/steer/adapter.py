"""Technique-gated low-rank adapters for the frozen Seed-VC DiT.

Each wrapped linear layer computes

    y = W x + B( g * (A x) ),   g = sum_k alpha_k * e_k  (one gate vector per layer)

A (rank x in) and B (out x rank) are shared by all techniques; e_k (rank) is a
learned gate per technique. With every alpha_k = 0 the adapter adds nothing and
is skipped entirely, so the base model is reproduced bit for bit. Each alpha_k
is a continuous control strength; several techniques can be set at once.
"""
from __future__ import annotations

import math

import torch
from torch import nn

DEFAULT_TECHNIQUES = ("breathy", "belt", "vocal_fry", "forte", "piano")
SKIP_NAME_PARTS = ("project_layer",)  # adaptive layer norm projections act on the time embedding


class _GateState:
    """Shared, mutable gate setting: alpha of shape [K] (all rows) or [B, K] (per row)."""

    def __init__(self, n_techniques):
        self.n = n_techniques
        self.alpha = None

    def set(self, alpha):
        if alpha is None:
            self.alpha = None
            return
        alpha = torch.as_tensor(alpha, dtype=torch.float32)
        if alpha.shape[-1] != self.n or alpha.dim() not in (1, 2):
            raise ValueError(f"alpha must have shape [{self.n}] or [B, {self.n}], got {tuple(alpha.shape)}")
        self.alpha = None if not bool(alpha.abs().gt(0).any()) else alpha


class GatedLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, state: _GateState, rank: int):
        super().__init__()
        self.base = base
        self.state = state
        self.rank = rank
        device = base.weight.device
        self.down = nn.Parameter(torch.empty(rank, base.in_features, device=device))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank, device=device))
        # Each technique starts on its own block of rank components (gate 1.0) and only weakly
        # on the others (0.1), so the techniques do not all start out as the same direction.
        gates = torch.full((state.n, rank), 0.1, device=device)
        block = max(1, rank // state.n)
        for k in range(state.n):
            gates[k, k * block:(k + 1) * block] = 1.0
        self.gates = nn.Parameter(gates)
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    @property
    def weight(self):  # some upstream code reads .weight (dtype / device lookups)
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        y = self.base(x)
        alpha = self.state.alpha
        if alpha is None:
            return y
        alpha = alpha.to(device=x.device, dtype=self.gates.dtype)
        gate = alpha @ self.gates  # [rank] or [B, rank]
        h = nn.functional.linear(x, self.down.to(x.dtype))
        if gate.dim() == 1:
            h = h * gate.to(h.dtype)
        else:
            if gate.size(0) != x.size(0):
                raise RuntimeError(f"per-row alpha has {gate.size(0)} rows, input batch is {x.size(0)}")
            h = h * gate.to(h.dtype).view(x.size(0), *([1] * (x.dim() - 2)), self.rank)
        return y + nn.functional.linear(h, self.up.to(h.dtype))


class TechniqueAdapter:
    """Installs gated adapters into a DiT estimator and manages them."""

    def __init__(self, estimator: nn.Module, techniques=DEFAULT_TECHNIQUES, rank: int = 40):
        self.estimator = estimator
        self.techniques = list(techniques)
        self.rank = rank
        self.state = _GateState(len(self.techniques))
        self.wrapped = {}
        targets = []
        for name, module in estimator.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not (name.startswith("transformer.layers.") or name.startswith("cond_x_merge_linear")
                    or name.startswith("final_mlp.")):
                continue
            if any(part in name for part in SKIP_NAME_PARTS):
                continue
            if name.endswith("skip_in_linear"):  # only the upper half of U-ViT layers receive a skip input
                layer = int(name.split(".")[2])
                if layer <= len(estimator.transformer.layers) // 2:
                    continue
            targets.append(name)
        for name in targets:
            parent_name, _, child = name.rpartition(".")
            parent = estimator.get_submodule(parent_name) if parent_name else estimator
            wrapper = GatedLoRALinear(getattr(parent, child), self.state, rank)
            setattr(parent, child, wrapper)
            self.wrapped[name] = wrapper
        if not self.wrapped:
            raise RuntimeError("no linear layers found to adapt")

    # ---- control -------------------------------------------------------
    def index(self, technique):
        return self.techniques.index(technique)

    def set(self, alpha=None, **amounts):
        """set(alpha_tensor) or set(breathy=0.8, vocal_fry=0.3); set() turns everything off."""
        if alpha is None and amounts:
            alpha = torch.zeros(len(self.techniques))
            for key, value in amounts.items():
                alpha[self.index(key)] = float(value)
        self.state.set(alpha)

    def off(self):
        self.state.set(None)

    # ---- parameters ----------------------------------------------------
    def parameters(self):
        for wrapper in self.wrapped.values():
            yield wrapper.down
            yield wrapper.up
            yield wrapper.gates

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def state_dict(self):
        return {"techniques": self.techniques, "rank": self.rank,
                "layers": {name: {"down": w.down.detach().cpu(), "up": w.up.detach().cpu(),
                                  "gates": w.gates.detach().cpu()} for name, w in self.wrapped.items()}}

    def load_state_dict(self, payload):
        if list(payload["techniques"]) != self.techniques or payload["rank"] != self.rank:
            raise ValueError("adapter file was trained with different techniques or rank")
        missing = set(self.wrapped) ^ set(payload["layers"])
        if missing:
            raise ValueError(f"adapter layer mismatch: {sorted(missing)[:5]}")
        with torch.no_grad():
            for name, wrapper in self.wrapped.items():
                for key in ("down", "up", "gates"):
                    getattr(wrapper, key).copy_(payload["layers"][name][key])

    @classmethod
    def from_file(cls, estimator, path):
        payload = torch.load(path, map_location="cpu")
        adapter = cls(estimator, payload["techniques"], payload["rank"])
        adapter.load_state_dict(payload)
        return adapter

    def remove(self):
        for name, wrapper in self.wrapped.items():
            parent_name, _, child = name.rpartition(".")
            parent = self.estimator.get_submodule(parent_name) if parent_name else self.estimator
            setattr(parent, child, wrapper.base)
        self.wrapped = {}
        self.state.set(None)
