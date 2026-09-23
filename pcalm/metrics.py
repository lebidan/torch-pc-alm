from __future__ import annotations

import torch


def mse_ce_accuracy(logits: torch.Tensor, y: torch.Tensor):
    mse = 0.5 * torch.mean(torch.sum((logits - y) ** 2, dim=-1))
    ce = -torch.mean(torch.sum(y * torch.log_softmax(logits, dim=-1), dim=-1))
    acc = torch.mean((torch.argmax(logits, dim=-1) == torch.argmax(y, dim=-1)).float())
    return mse, ce, acc


def tree_l2(tree) -> torch.Tensor:
    leaves = list(tree)
    if not leaves:
        return torch.tensor(0.0)
    return torch.sqrt(sum(torch.sum(x * x) for x in leaves))


def tree_dot(a, b) -> torch.Tensor:
    pairs = list(zip(a, b))
    if not pairs:
        return torch.tensor(0.0)
    return sum(torch.sum(x * y) for x, y in pairs)


def tree_cos(a, b) -> torch.Tensor:
    return tree_dot(a, b) / torch.clamp(tree_l2(a) * tree_l2(b), min=1e-30)
