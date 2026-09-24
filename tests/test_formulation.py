from __future__ import annotations

import pytest
import torch

from pcalm.config import MethodConfig
from pcalm.inference import infer, method_loss
from pcalm.model import ResidualMLP
from pcalm.training import adam_learning_rate


def small_case():
    model = ResidualMLP(seed=0, depth=4, width=5, input_dim=3, output_dim=2, activation="tanh")
    x = torch.randn((7, 3), generator=torch.Generator().manual_seed(1))
    y = torch.nn.functional.one_hot(torch.arange(7) % 2, 2).float()
    return model, x, y


def pcalm(**kw):
    return MethodConfig(**{"name": "pcalm", "budget": 3, "state_lr": 0.1, "rho": 1.0, **kw})


def test_constraints_are_hidden_edges_only():
    model, x, _ = small_case()
    z = model.feedforward_hidden(x)
    residuals = model.constraint_residuals(x, z)
    assert residuals.shape == (3, x.shape[0], 5)
    torch.testing.assert_close(residuals, torch.zeros_like(residuals), atol=1e-6, rtol=0)


def test_pc_has_zero_duals():
    model, x, y = small_case()
    _, duals = infer(model, x, y, MethodConfig(name="pc", budget=2, state_lr=0.1))
    assert torch.count_nonzero(duals) == 0


def test_pcalm_alpha_zero_matches_pc_gradient():
    model, x, y = small_case()
    g_pc = torch.autograd.grad(method_loss(model, x, y, pcalm(name="pc")), list(model.parameters()))
    g_alm = torch.autograd.grad(method_loss(model, x, y, pcalm(alpha=0.0)), list(model.parameters()))
    for a, b in zip(g_pc, g_alm):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_pcalm_duals_update_hidden_edges():
    model, x, y = small_case()
    _, duals = infer(model, x, y, pcalm(budget=2, weight_credit_timing="post_dual_energy"))
    assert duals.shape == (3, x.shape[0], 5)
    assert torch.count_nonzero(duals) > 0


def test_default_adam_lr_uses_width_depth_scaling():
    assert adam_learning_rate(width=64, depth=16, eta0=1e-3, gamma0=1.0, explicit_lr=None) == 2e-3
    assert adam_learning_rate(width=64, depth=16, eta0=1e-3, gamma0=1.0, explicit_lr=5e-4) == 5e-4


def test_inference_is_per_sample_batch_invariant():
    model, x, y = small_case()
    z_single, _ = infer(model, x[:1], y[:1], pcalm(alpha=1.0))
    z_batch, _ = infer(model, x, y, pcalm(alpha=1.0))
    torch.testing.assert_close(z_single[:, 0], z_batch[:, 0], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("bad", [dict(name="sgd"), dict(rho=0.0), dict(budget=0), dict(inner_steps=0),
                                 dict(weight_credit_timing="later")])
def test_invalid_method_config_is_rejected(bad):
    with pytest.raises(ValueError):
        pcalm(**bad)
