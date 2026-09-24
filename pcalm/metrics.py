from __future__ import annotations

import torch
import torch.nn.functional as F


def mse(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Training loss: 0.5 * batch mean of the squared error summed over classes."""
    return F.mse_loss(out, y, reduction="sum") / (2 * y.shape[0])


def mse_ce_accuracy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[MSE, cross-entropy, accuracy] for one-hot targets `y`; CE is reported, never trained on."""
    acc = (logits.argmax(-1) == y.argmax(-1)).float().mean()
    return torch.stack((mse(logits, y), F.cross_entropy(logits, y), acc))


def grad_cosine(a, b) -> torch.Tensor:
    """Cosine between two gradients, each flattened over all weight tensors."""
    a, b = (torch.cat([g.flatten() for g in grads]) for grads in (a, b))
    return F.cosine_similarity(a, b, dim=0)
