"""Scalar-decay QGDN. See research/qgdn/DERIVATION_PACKAGE.md, §8.2–8.7."""
import math

import torch
from torch import nn

from .gdn import GatedDeltaNet


class QueryGuidedDeltaNet(GatedDeltaNet):
    def __init__(self, *args, recall_mode="query", recall_gate="token", recall_init=0.1,
                 recall_weight_init="zero", **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_groups != self.num_heads or self.use_lsa:
            raise ValueError("The Recall study requires independent MHA states; GQA/LSA is not supported.")
        if self.allow_neg_eigval:
            raise ValueError("QGDN assumes beta in [0,1]; allow_neg_eigval must be False.")
        if recall_mode not in {"query", "key", "isotropic"}:
            raise ValueError(f"Unknown recall_mode: {recall_mode}")
        if recall_gate not in {"token", "head", "fixed"}:
            raise ValueError(f"Unknown recall_gate: {recall_gate}")
        if recall_weight_init not in {"zero", "beta"}:
            raise ValueError(f"Unknown recall_weight_init: {recall_weight_init}")
        if recall_weight_init == "beta" and recall_gate != "token":
            raise ValueError("beta weight initialization requires a token-dependent projection")
        if not 0 <= recall_init <= 1 or (recall_gate != "fixed" and not 0 < recall_init < 1):
            raise ValueError("Learned gates require 0 < recall_init < 1; fixed gates allow endpoints.")
        self.recall_mode, self.recall_gate, self.recall_init = recall_mode, recall_gate, recall_init
        self.recall_weight_init = recall_weight_init
        if recall_gate == "token":
            # Do not change the RNG stream used to initialize shared backbone weights.
            with torch.random.fork_rng(devices=[]):
                self.recall_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
                if recall_weight_init == "beta":
                    nn.init.xavier_uniform_(self.recall_proj.weight, gain=2**-2.5)
                else:
                    nn.init.zeros_(self.recall_proj.weight)
            nn.init.constant_(self.recall_proj.bias, math.log(recall_init / (1 - recall_init)))
            self.recall_proj.bias._no_reinit = True
        elif recall_gate == "head":
            self.recall_logit = nn.Parameter(torch.full((self.num_heads,), math.log(recall_init / (1 - recall_init))))
            self.recall_logit._no_weight_decay = True
        self.collect_recall_stats = False
        self.last_recall_stats = None

    def recall_gamma(self, x):
        if self.recall_gate == "token":
            gamma = self.recall_proj(x).float().sigmoid()
        elif self.recall_gate == "head":
            gamma = self.recall_logit.float().sigmoid().expand(*x.shape[:2], self.num_heads)
        else:
            gamma = x.new_full((*x.shape[:2], self.num_heads), self.recall_init, dtype=torch.float32)
        if self.collect_recall_stats:
            with torch.no_grad():
                g = -self.A_log.float().exp() * torch.nn.functional.softplus(self.gk_proj(x).float() + self.dt_bias)
                alpha = g.exp()
                margin = (-g.expm1()) * (1 - gamma)
                self.last_recall_stats = torch.stack((gamma.mean(), (gamma > 0.95).float().mean(),
                                                     alpha.mean(), margin.mean())).detach()
        return gamma
