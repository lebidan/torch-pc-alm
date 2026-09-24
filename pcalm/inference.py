"""BP, PC, and PC-ALM weight gradients, all obtained by autograd.

PC and PC-ALM relax the hidden states `z` (the free variables `free` of the
original JAX code) and, for PC-ALM, the duals `lam`, with the weights held fixed, then take the weight gradient of the energy at
the final, detached `z` and `lam`. No gradient flows through the relaxation.
"""

from __future__ import annotations

import torch

from .config import MethodConfig
from .metrics import mse
from .model import ResidualMLP


def energy(model: ResidualMLP, x, y, z, lam, rho: float) -> torch.Tensor:
    """Shifted augmented energy: mse + rho / (2B) * sum_l ||c_l + lam_l / rho||^2.

    The output edge is supervised and has no constraint or dual.
    """
    c = model.constraint_residuals(x, z)
    return mse(model.readout(z[-1]), y) + rho / (2 * x.shape[0]) * (c + lam / rho).square().sum()


# Set TORCH_COMPILE_DISABLE=1 to run eagerly (debugging, or no Triton/C++ compiler).
@torch.compile
def _outer_step(model, x, y, z, lam, *, state_lr, rho, alpha, inner_steps, update_dual):
    # The energy is a batch mean, so scaling the step by the batch size makes
    # each sample's dynamics independent of the batch size.
    lr = state_lr * x.shape[0]
    for _ in range(inner_steps):
        z = z - lr * torch.func.grad(energy, argnums=3)(model, x, y, z, lam, rho)
    if update_dual:
        lam = lam + alpha * model.constraint_residuals(x, z)
    return z, lam


@torch.no_grad()
def relax(model: ResidualMLP, x, y, method: MethodConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Run PC or PC-ALM inference from the feed-forward states.

    PC is `budget` activity steps with zero duals. PC-ALM runs `budget` outer
    iterations of `inner_steps` activity steps, each followed by the dual
    update `lam += alpha * c`. The final update is kept only for
    `post_dual_energy` weight credit.
    """
    pcalm = method.name == "pcalm"
    z = model.feedforward_hidden(x)  # initial guess; `free_init` in the JAX code
    lam = torch.zeros_like(z)
    for t in range(method.budget):
        last = t == method.budget - 1
        z, lam = _outer_step(
            model, x, y, z, lam, state_lr=method.state_lr, rho=method.rho, alpha=method.alpha,
            inner_steps=method.inner_steps if pcalm else 1,
            update_dual=pcalm and (not last or method.weight_credit_timing == "post_dual_energy"),
        )
    return z, lam


def method_loss(model: ResidualMLP, x, y, method: MethodConfig) -> torch.Tensor:
    """Scalar whose weight gradient is the method's update direction."""
    if method.name == "bp":
        return mse(model(x), y)
    z, lam = relax(model, x, y, method)
    return energy(model, x, y, z, lam, method.rho)
