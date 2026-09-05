"""Q-Delta mixer with a learned per-token, per-head query-feedback gate."""
from __future__ import annotations

import torch
from torch import nn

from .gdn import GatedDeltaNet


class QueryDeltaNet(GatedDeltaNet):
    """Paper Q-Delta on the shared 340M GDN backbone.

    The released implementation parameterizes
    ``lambda_t = sigmoid(W_lambda h_t - lambda_bias)``.  Constructing and
    initializing the extra projection inside ``fork_rng`` keeps every shared
    GDN parameter bit-identical for paired seeded experiments.
    """

    def __init__(
        self, *args, qdelta_lambda_bias: float = 0.9,
        qdelta_query_sign: float = 1.0, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.num_groups != self.num_heads or self.use_lsa:
            raise ValueError("The Q-Delta comparison requires independent MHA states; GQA/LSA is unsupported.")
        if self.allow_neg_eigval:
            raise ValueError("Q-Delta assumes beta in [0,1]; allow_neg_eigval must be False.")
        if qdelta_lambda_bias < 0:
            raise ValueError("qdelta_lambda_bias must be non-negative")
        if qdelta_query_sign not in {-1.0, 1.0}:
            raise ValueError("qdelta_query_sign must be +1 or -1")
        self.update_rule = "qdelta"
        self.qdelta_lambda_bias = float(qdelta_lambda_bias)
        self.qdelta_query_sign = float(qdelta_query_sign)
        with torch.random.fork_rng(devices=[]):
            self.lambda_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
            nn.init.xavier_uniform_(self.lambda_proj.weight, gain=2**-2.5)

    def query_feedback_lambda(self, x: torch.Tensor) -> torch.Tensor:
        return (self.lambda_proj(x).float() - self.qdelta_lambda_bias).sigmoid()
