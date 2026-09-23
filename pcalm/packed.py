"""Batched hidden-layer PC/PC-ALM updates for the training path.

Every activity, dual, and weight-gradient operation is explicit. This avoids
building an autograd graph across the inner loop's many small layers on CUDA.
"""

from __future__ import annotations

import torch

from .model import activation_derivative, forward


def method_grad_packed(params, scales, skips, x, y, schedule, *, state_lr: float, rho: float, phi) -> list[torch.Tensor]:
    if rho <= 0:
        raise ValueError("rho must be positive")
    if schedule.family == "pcalm":
        if schedule.budget < 1:
            raise ValueError("PC-ALM budget must be at least 1")
        if schedule.inner_steps < 1:
            raise ValueError("PC-ALM inner_steps must be at least 1")
        if schedule.weight_credit_timing not in {"pre_dual_energy", "post_dual_energy"}:
            raise ValueError("weight_credit_timing must be pre_dual_energy or post_dual_energy")
    elif schedule.family != "pc":
        raise ValueError(f"unknown schedule family: {schedule.family}")

    with torch.no_grad():
        weights = [p.detach() for p in params]
        states = torch.stack(forward(weights, scales, skips, x, phi)[:-1])
        duals = torch.zeros_like(states)
        middle_weights = torch.stack(weights[1:-1]) if len(weights) > 2 else None
        middle_scale = scales[1] if middle_weights is not None else 0.0

        def residuals(z):
            first = z[0] - scales[0] * (x @ weights[0].T)
            if middle_weights is None:
                return first.unsqueeze(0)
            middle_pred = z[:-1] + middle_scale * torch.bmm(phi(z[:-1]), middle_weights.transpose(1, 2))
            return torch.cat((first.unsqueeze(0), z[1:] - middle_pred), dim=0)

        def activity_step(z, lam):
            c = residuals(z)
            q = rho * c + lam
            output_error = scales[-1] * (phi(z[-1]) @ weights[-1].T) - y
            g = q.clone()
            if middle_weights is not None:
                g[:-1] -= q[1:] + middle_scale * torch.bmm(q[1:], middle_weights) * activation_derivative(phi, z[:-1])
            g[-1] += scales[-1] * (output_error @ weights[-1]) * activation_derivative(phi, z[-1])
            return z - state_lr * g

        if schedule.family == "pc":
            for _ in range(max(schedule.budget, 0)):
                states = activity_step(states, duals)
        else:
            for outer in range(schedule.budget):
                for _ in range(schedule.inner_steps):
                    states = activity_step(states, duals)
                if outer < schedule.budget - 1 or schedule.weight_credit_timing == "post_dual_energy":
                    duals = duals + schedule.alpha * residuals(states)

        c = residuals(states)
        q = rho * c + duals
        batch_size = x.shape[0]
        gradients = [-scales[0] * (q[0].T @ x) / batch_size]
        if middle_weights is not None:
            middle_grad = -middle_scale * torch.bmm(q[1:].transpose(1, 2), phi(states[:-1])) / batch_size
            gradients.extend(middle_grad.unbind(0))
        output_error = scales[-1] * (phi(states[-1]) @ weights[-1].T) - y
        gradients.append(scales[-1] * (output_error.T @ phi(states[-1])) / batch_size)
        return gradients
