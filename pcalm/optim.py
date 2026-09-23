"""Optimizer settings matching the original JAX Adam update."""

from __future__ import annotations

import torch


def make_adam(params, learning_rate: float) -> torch.optim.Adam:
    return torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
