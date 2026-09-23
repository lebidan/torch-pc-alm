from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import Params, block_pred, forward
from .packed import method_grad_packed


@dataclass(frozen=True)
class Schedule:
    family: str
    budget: int
    alpha: float = 0.0
    inner_steps: int = 1
    weight_credit_timing: str = "pre_dual_energy"


def supervised_loss(params: Params, scales, skips, x, y, free, phi) -> torch.Tensor:
    y_pred = block_pred(params[-1], scales[-1], skips[-1], free[-1], phi, is_first=False)
    return 0.5 * torch.mean(torch.sum((y_pred - y) ** 2, dim=-1))


def bp_loss(params: Params, scales, skips, x, y, phi) -> torch.Tensor:
    y_pred = forward(params, scales, skips, x, phi)[-1]
    return 0.5 * torch.mean(torch.sum((y_pred - y) ** 2, dim=-1))


def free_init(params: Params, scales, skips, x, phi) -> list[torch.Tensor]:
    return forward(params, scales, skips, x, phi)[:-1]


def constraint_residuals(params: Params, scales, skips, x, free, phi) -> list[torch.Tensor]:
    """Only hidden edges are constraints; the output edge is supervised."""
    residuals = []
    for layer_ix, z_l in enumerate(free):
        z_prev = x if layer_ix == 0 else free[layer_ix - 1]
        pred = block_pred(params[layer_ix], scales[layer_ix], skips[layer_ix], z_prev, phi, is_first=(layer_ix == 0))
        residuals.append(z_l - pred)
    return residuals


def zero_duals_like(residuals: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.zeros_like(c) for c in residuals]


def al_energy_shifted(params: Params, scales, skips, x, y, free, duals, rho: float, phi) -> torch.Tensor:
    if rho <= 0:
        raise ValueError("rho must be positive")
    residuals = constraint_residuals(params, scales, skips, x, free, phi)
    total = supervised_loss(params, scales, skips, x, y, free, phi)
    batch_size = x.shape[0]
    for residual, dual in zip(residuals, duals):
        shifted = residual + dual / rho
        total = total + 0.5 * rho * torch.sum(shifted * shifted) / batch_size
    return total


def _detached_params(params: Params) -> list[torch.Tensor]:
    return [p.detach() for p in params]


def run_pc(params: Params, scales, skips, x, y, *, state_lr: float, rho: float, steps: int, phi):
    if rho <= 0:
        raise ValueError("rho must be positive")
    fixed_params = _detached_params(params)
    with torch.no_grad():
        free0 = free_init(fixed_params, scales, skips, x, phi)
        duals0 = zero_duals_like(constraint_residuals(fixed_params, scales, skips, x, free0, phi))
    return _solve_inner(fixed_params, scales, skips, x, y, free0, duals0, state_lr, rho, steps, phi), duals0


def run_pcalm(
    params: Params, scales, skips, x, y, *, state_lr: float, rho: float, alpha: float,
    budget: int, inner_steps: int, weight_credit_timing: str, phi,
):
    if rho <= 0:
        raise ValueError("rho must be positive")
    if budget < 1:
        raise ValueError("PC-ALM budget must be at least 1")
    if inner_steps < 1:
        raise ValueError("PC-ALM inner_steps must be at least 1")
    if weight_credit_timing not in {"pre_dual_energy", "post_dual_energy"}:
        raise ValueError("weight_credit_timing must be pre_dual_energy or post_dual_energy")

    fixed_params = _detached_params(params)
    with torch.no_grad():
        free = free_init(fixed_params, scales, skips, x, phi)
        duals = zero_duals_like(constraint_residuals(fixed_params, scales, skips, x, free, phi))
    for _ in range(budget - 1):
        free = _solve_inner(fixed_params, scales, skips, x, y, free, duals, state_lr, rho, inner_steps, phi)
        with torch.no_grad():
            residuals = constraint_residuals(fixed_params, scales, skips, x, free, phi)
            duals = [lam + alpha * r for lam, r in zip(duals, residuals)]

    free = _solve_inner(fixed_params, scales, skips, x, y, free, duals, state_lr, rho, inner_steps, phi)
    with torch.no_grad():
        residuals = constraint_residuals(fixed_params, scales, skips, x, free, phi)
        duals_after = [lam + alpha * r for lam, r in zip(duals, residuals)]
    return free, duals if weight_credit_timing == "pre_dual_energy" else duals_after


def infer_for_schedule(params: Params, scales, skips, x, y, schedule: Schedule, *, state_lr: float, rho: float, phi):
    if schedule.family == "pc":
        return run_pc(params, scales, skips, x, y, state_lr=state_lr, rho=rho, steps=schedule.budget, phi=phi)
    if schedule.family == "pcalm":
        return run_pcalm(params, scales, skips, x, y, state_lr=state_lr, rho=rho, alpha=schedule.alpha,
                         budget=schedule.budget, inner_steps=schedule.inner_steps,
                         weight_credit_timing=schedule.weight_credit_timing, phi=phi)
    raise ValueError(f"unknown schedule family: {schedule.family}")


def method_grad(params: Params, scales, skips, x, y, schedule: Schedule, *, state_lr: float, rho: float, phi):
    if schedule.family in {"pc", "pcalm"}:
        return method_grad_packed(params, scales, skips, x, y, schedule,
                                  state_lr=state_lr, rho=rho, phi=phi)
    return _method_grad_autograd(params, scales, skips, x, y, schedule,
                                 state_lr=state_lr, rho=rho, phi=phi)


def _method_grad_autograd(params: Params, scales, skips, x, y, schedule: Schedule, *, state_lr: float, rho: float, phi):
    with torch.enable_grad():
        if schedule.family == "bp":
            loss = bp_loss(params, scales, skips, x, y, phi)
        else:
            free, duals = infer_for_schedule(params, scales, skips, x, y, schedule,
                                             state_lr=state_lr, rho=rho, phi=phi)
            loss = al_energy_shifted(params, scales, skips, x, y, free, duals, rho, phi)
        return list(torch.autograd.grad(loss, list(params)))


def _solve_inner(params: Params, scales, skips, x, y, free, duals, state_lr: float, rho: float, steps: int, phi):
    if steps <= 0:
        return [z.detach() for z in free]
    effective_lr = state_lr * free[0].shape[0]
    for _ in range(steps):
        with torch.enable_grad():
            free_vars = [z.detach().requires_grad_(True) for z in free]
            energy = al_energy_shifted(params, scales, skips, x, y, free_vars, duals, rho, phi)
            grads = torch.autograd.grad(energy, free_vars)
        free = [(z - effective_lr * g).detach() for z, g in zip(free_vars, grads)]
    return free
