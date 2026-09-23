from __future__ import annotations

import torch

from pcalm.inference import Schedule, constraint_residuals, method_grad, run_pc, run_pcalm
from pcalm.model import activation_fn, init_params, model_scales, skip_mask
from pcalm.training import adam_learning_rate


def small_case():
    params = init_params(0, depth=4, width=5, input_dim=3, output_dim=2)
    scales = model_scales(width=5, depth=4, input_dim=3)
    skips = skip_mask(4)
    phi = activation_fn("tanh")
    x = torch.randn((7, 3), generator=torch.Generator().manual_seed(1))
    y = torch.nn.functional.one_hot(torch.arange(7) % 2, 2).float()
    return params, scales, skips, phi, x, y


def test_constraints_are_hidden_edges_only():
    params, scales, skips, phi, x, _ = small_case()
    free = [torch.zeros((x.shape[0], 5)) for _ in range(3)]
    residuals = constraint_residuals(params, scales, skips, x, free, phi)
    assert len(residuals) == len(params) - 1
    assert all(r.shape == (x.shape[0], 5) for r in residuals)


def test_pc_has_zero_duals():
    params, scales, skips, phi, x, y = small_case()
    _, duals = run_pc(params, scales, skips, x, y, state_lr=0.1, rho=1.0, steps=2, phi=phi)
    assert all(torch.allclose(dual, torch.zeros_like(dual)) for dual in duals)


def test_pcalm_alpha_zero_matches_pc_gradient():
    params, scales, skips, phi, x, y = small_case()
    pc_schedule = Schedule(family="pc", budget=3)
    alm_schedule = Schedule(family="pcalm", budget=3, alpha=0.0)
    g_pc = method_grad(params, scales, skips, x, y, pc_schedule, state_lr=0.1, rho=1.0, phi=phi)
    g_alm = method_grad(params, scales, skips, x, y, alm_schedule, state_lr=0.1, rho=1.0, phi=phi)
    for a, b in zip(g_pc, g_alm):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_pcalm_duals_update_hidden_edges():
    params, scales, skips, phi, x, y = small_case()
    _, duals = run_pcalm(params, scales, skips, x, y, state_lr=0.1, rho=1.0,
                         alpha=1.0, budget=2, inner_steps=1,
                         weight_credit_timing="post_dual_energy", phi=phi)
    assert len(duals) == len(params) - 1
    assert any(float(torch.linalg.norm(dual)) > 0.0 for dual in duals)


def test_default_adam_lr_uses_width_depth_scaling():
    assert adam_learning_rate(width=64, depth=16, eta0=1e-3, gamma0=1.0, explicit_lr=None) == 2e-3
    assert adam_learning_rate(width=64, depth=16, eta0=1e-3, gamma0=1.0, explicit_lr=5e-4) == 5e-4


def test_inference_is_per_sample_batch_invariant():
    params, scales, skips, phi, x, y = small_case()
    kw = dict(state_lr=0.1, rho=1.0, alpha=1.0, budget=3, inner_steps=1,
              weight_credit_timing="pre_dual_energy", phi=phi)
    free_single, _ = run_pcalm(params, scales, skips, x[:1], y[:1], **kw)
    free_batch, _ = run_pcalm(params, scales, skips, x, y, **kw)
    for a, b in zip(free_single, free_batch):
        assert torch.allclose(a[0], b[0], atol=1e-5, rtol=1e-5)
