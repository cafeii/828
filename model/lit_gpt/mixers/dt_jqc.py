"""Model wrappers for the DT-GDN and JQC-GDN mechanisms."""
from __future__ import annotations

import math

import torch
from torch import nn

from .gdn import GatedDeltaNet


class _QueryConsolidationNet(GatedDeltaNet):
    method: str

    def __init__(self, *args, recall_init=0.1, recall_gate="token", **kwargs):
        # These methods intentionally have one shared input-dependent gamma gate.
        kwargs.pop("recall_mode", None)
        kwargs.pop("recall_weight_init", None)
        super().__init__(*args, **kwargs)
        if self.num_groups != self.num_heads or self.use_lsa:
            raise ValueError("DT/JQC experiments require independent MHA states")
        if self.allow_neg_eigval:
            raise ValueError("DT/JQC require beta in [0,1]")
        if recall_gate != "token":
            raise ValueError("DT/JQC formal models require an input-dependent token gamma gate")
        if not 0 < recall_init < 1:
            raise ValueError("A learned gamma gate requires 0 < recall_init < 1")
        self.recall_mode = self.method
        self.recall_gate = recall_gate
        self.recall_init = recall_init
        # Preserve the random stream used by every shared GDN parameter.
        with torch.random.fork_rng(devices=[]):
            self.recall_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
            nn.init.zeros_(self.recall_proj.weight)
        nn.init.constant_(self.recall_proj.bias, math.log(recall_init / (1 - recall_init)))
        self.recall_proj.bias._no_reinit = True

    def recall_gamma(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.recall_proj(hidden_states).float().sigmoid()


class DualTargetGatedDeltaNet(_QueryConsolidationNet):
    """Joint proximal write and query-recall targets (DT-GDN)."""

    method = "dt"


class JointQueryConsolidationGatedDeltaNet(_QueryConsolidationNet):
    """Native GDN write followed by query consolidation (JQC-GDN)."""

    method = "jqc"
