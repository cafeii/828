"""QGDN model wrapper and recall-gate parameterization."""
from __future__ import annotations

import math

import torch
from torch import nn

from .gdn import GatedDeltaNet


class QueryGuidedDeltaNet(GatedDeltaNet):
    """GDN with configurable ordering of Recall and Delta corrections."""

    def __init__(self, *args, recall_mode="query", recall_order="recall_then_delta",
                 recall_gate="token", recall_init=0.5,
                 recall_weight_init="beta", recall_uniform_min=0.85,
                 recall_uniform_max=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_groups != self.num_heads or self.use_lsa:
            raise ValueError("The Recall study requires independent MHA states; GQA/LSA is not supported.")
        if self.allow_neg_eigval:
            raise ValueError("QGDN assumes beta in [0,1]; allow_neg_eigval must be False.")
        if recall_mode not in {"query", "key", "isotropic"}:
            raise ValueError(f"Unknown recall_mode: {recall_mode}")
        if recall_order not in {"recall_then_delta", "delta_then_recall", "parallel"}:
            raise ValueError(f"Unknown recall_order: {recall_order}")
        if recall_mode == "isotropic" and recall_order != "recall_then_delta":
            raise ValueError("The isotropic control only defines recall_then_delta ordering")
        if recall_gate not in {"token", "head", "fixed"}:
            raise ValueError(f"Unknown recall_gate: {recall_gate}")
        if recall_weight_init not in {"zero", "beta", "uniform_gate"}:
            raise ValueError(f"Unknown recall_weight_init: {recall_weight_init}")
        if not 0 <= recall_init <= 1 or (recall_gate != "fixed" and not 0 < recall_init < 1):
            raise ValueError("Learned gates require 0 < recall_init < 1; fixed gates allow endpoints.")
        if recall_gate == "token" and recall_weight_init == "beta" and recall_init != 0.5:
            raise ValueError("beta-style gamma initialization requires zero bias (recall_init=0.5)")
        if recall_weight_init == "uniform_gate":
            if recall_gate != "token":
                raise ValueError("uniform-gate gamma initialization requires a token-wise gate")
            if not 0 < recall_uniform_min < recall_uniform_max < 1:
                raise ValueError("uniform-gate gamma bounds must satisfy 0 < min < max < 1")
        self.recall_mode = recall_mode
        self.recall_order = recall_order
        self.recall_gate, self.recall_init = recall_gate, recall_init
        self.recall_weight_init = recall_weight_init
        self.recall_uniform_min = recall_uniform_min
        self.recall_uniform_max = recall_uniform_max
        if recall_gate == "token":
            # Do not change the RNG stream used to initialize shared backbone weights.
            with torch.random.fork_rng(devices=[]):
                self.recall_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
                if recall_weight_init == "beta":
                    # Match b_proj's initialization scheme without tying the two gates:
                    # independent Xavier-uniform weights with the same gain.
                    nn.init.xavier_uniform_(self.recall_proj.weight, gain=2**-2.5)
                    nn.init.zeros_(self.recall_proj.bias)
                elif recall_weight_init == "zero":
                    nn.init.zeros_(self.recall_proj.weight)
                    nn.init.constant_(self.recall_proj.bias, math.log(recall_init / (1 - recall_init)))
                else:
                    # Sample in gate space so every initial gamma is guaranteed to
                    # lie in the requested interval.  Zero projection weights make
                    # the initial value input-independent, but remain trainable.
                    nn.init.zeros_(self.recall_proj.weight)
                    initial_gamma = torch.empty_like(self.recall_proj.bias).uniform_(
                        recall_uniform_min, recall_uniform_max
                    )
                    with torch.no_grad():
                        self.recall_proj.bias.copy_(torch.logit(initial_gamma))
            self.recall_proj.bias._no_reinit = True
        elif recall_gate == "head":
            self.recall_logit = nn.Parameter(torch.full((self.num_heads,), math.log(recall_init / (1 - recall_init))))
            self.recall_logit._no_weight_decay = True

    def recall_gamma(self, x: torch.Tensor) -> torch.Tensor:
        if self.recall_gate == "token":
            gamma = self.recall_proj(x).float().sigmoid()
        elif self.recall_gate == "head":
            gamma = self.recall_logit.float().sigmoid().expand(*x.shape[:2], self.num_heads)
        else:
            gamma = x.new_full((*x.shape[:2], self.num_heads), self.recall_init, dtype=torch.float32)
        return gamma
