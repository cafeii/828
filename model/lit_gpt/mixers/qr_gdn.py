"""Model wrapper for the two-state Query-Recall Gated Delta Net."""
from __future__ import annotations

import math

import torch
from einops import repeat
from torch import nn
from torch.nn import functional as F

from .gdn import GatedDeltaNet


class QueryRecallGatedDeltaNet(GatedDeltaNet):
    """GDN KV memory plus a same-sized query-to-recall memory."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("recall_mode", None)
        kwargs.pop("recall_gate", None)
        kwargs.pop("recall_init", None)
        kwargs.pop("recall_weight_init", None)
        super().__init__(*args, **kwargs)
        if self.num_groups != self.num_heads or self.use_lsa:
            raise ValueError("QR-GDN formal comparison requires independent MHA states")
        if self.allow_neg_eigval:
            raise ValueError("QR-GDN requires beta gates in [0,1]")
        self.recall_mode = "qr_gdn"

        # Preserve the random stream of all parameters shared with native GDN,
        # including modules constructed after this mixer inside the full GPT.
        with torch.random.fork_rng(devices=[]):
            self.qr_gk_proj = nn.Linear(self.hidden_size, self.num_groups, bias=False)
            self.qr_b_proj = nn.Linear(self.hidden_size, self.num_groups, bias=True)
            self.qr_read_proj = nn.Linear(self.hidden_size, self.num_groups, bias=True)
            nn.init.xavier_uniform_(self.qr_gk_proj.weight, gain=2**-2.5)
            nn.init.xavier_uniform_(self.qr_b_proj.weight, gain=2**-2.5)
            nn.init.zeros_(self.qr_b_proj.bias)
            nn.init.zeros_(self.qr_read_proj.weight)
            nn.init.zeros_(self.qr_read_proj.bias)

            self.qr_A_log = nn.Parameter(
                torch.log(torch.empty(self.num_groups, dtype=torch.float32).uniform_(0, 16))
            )
            qr_dt = torch.exp(
                torch.rand(self.num_groups, dtype=torch.float32)
                * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            ).clamp(min=1e-4)
            self.qr_dt_bias = nn.Parameter(qr_dt + torch.log(-torch.expm1(-qr_dt)))

        self.qr_A_log._no_weight_decay = True
        self.qr_dt_bias._no_weight_decay = True
        self.qr_b_proj.bias._no_reinit = True
        self.qr_read_proj.bias._no_reinit = True

    def qr_gate_values(self, hidden_states: torch.Tensor):
        """Return log alpha^QR, beta^QR and the bounded-read preactivation."""
        g_qr = -self.qr_A_log.float().exp() * F.softplus(
            self.qr_gk_proj(hidden_states).float() + self.qr_dt_bias
        )
        b_qr = self.qr_b_proj(hidden_states).float().sigmoid()
        read_logit = self.qr_read_proj(hidden_states).float()
        if self.heads_per_group > 1:
            g_qr, b_qr, read_logit = (
                repeat(x, "... g -> ... (g i)", i=self.heads_per_group)
                for x in (g_qr, b_qr, read_logit)
            )
        return g_qr, b_qr, read_logit
