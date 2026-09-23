from __future__ import annotations

import csv
import json
import math
import pytest
import torch

from pcalm.config import ExperimentConfig, MethodConfig, ModelConfig, TrainingConfig
from pcalm.inference import Schedule, _method_grad_autograd, method_grad, run_pcalm
from pcalm.model import ResidualMLP
from pcalm.training import train_one


@pytest.mark.parametrize("method", ["bp", "pc", "pcalm"])
def test_cpu_smoke_and_output_schema(tmp_path, method):
    config = ExperimentConfig(
        dataset="synthetic", device="cpu", output_dir=str(tmp_path / method),
        model=ModelConfig(width=4, depth=3, input_dim=12, output_dim=3),
        method=MethodConfig(name=method, budget=2, state_lr=0.1),
        training=TrainingConfig(seed=2, epochs=1, batch_size=4, train_subset=8, test_subset=5),
    )
    summary = train_one(config)
    assert summary["steps"] == 2
    assert all(math.isfinite(summary[k]) for k in ("final_train_acc", "final_test_acc", "grad_cos_to_bp"))
    assert (tmp_path / method / "metrics.csv").is_file()
    assert json.loads((tmp_path / method / "summary.json").read_text()) == summary
    assert len(list(csv.DictReader((tmp_path / method / "metrics.csv").open()))) == 1


def test_depth_two_and_single_outer_step():
    model = ResidualMLP(seed=0, depth=2, width=3, input_dim=2, output_dim=2,
                        activation="linear", device="cpu")
    x = torch.tensor([[0.2, 0.4]], dtype=torch.float32)
    y = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    free, duals = run_pcalm(model.weights, model.scales, model.skips, x, y,
                            state_lr=0.1, rho=1.0, alpha=1.0, budget=1,
                            inner_steps=2, weight_credit_timing="post_dual_energy", phi=model.phi)
    assert len(free) == len(duals) == 1
    assert free[0].shape == (1, 3)
    assert not free[0].requires_grad
    for schedule in (Schedule("pc", 0), Schedule("pcalm", 1, 1.0, 2, "pre_dual_energy"),
                     Schedule("pcalm", 1, 1.0, 2, "post_dual_energy")):
        fast = method_grad(model.weights, model.scales, model.skips, x, y, schedule,
                           state_lr=0.1, rho=1.0, phi=model.phi)
        reference = _method_grad_autograd(model.weights, model.scales, model.skips, x, y,
                                          schedule, state_lr=0.1, rho=1.0, phi=model.phi)
        for a, b in zip(fast, reference):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible in this environment")
@pytest.mark.parametrize("method", ["bp", "pc", "pcalm"])
def test_cuda_smoke(tmp_path, method):
    config = ExperimentConfig(
        dataset="synthetic", device="cuda", output_dir=str(tmp_path / method),
        model=ModelConfig(width=4, depth=3, input_dim=12, output_dim=3),
        method=MethodConfig(name=method, budget=2, state_lr=0.1),
        training=TrainingConfig(seed=2, epochs=1, batch_size=4, train_subset=8, test_subset=5),
    )
    summary = train_one(config)
    assert math.isfinite(summary["final_test_acc"])
    model = ResidualMLP(seed=0, depth=3, width=4, input_dim=12, output_dim=3,
                        activation="relu", device="cuda")
    assert all(p.is_cuda for p in model.parameters())
    x = torch.randn(2, 12, device="cuda")
    y = torch.nn.functional.one_hot(torch.tensor([0, 1], device="cuda"), 3).float()
    free, duals = run_pcalm(model.weights, model.scales, model.skips, x, y,
                            state_lr=0.1, rho=1.0, alpha=1.0, budget=2,
                            inner_steps=1, weight_credit_timing="pre_dual_energy", phi=model.phi)
    assert all(t.is_cuda for t in [*free, *duals])
    grads = method_grad(model.weights, model.scales, model.skips, x, y,
                        Schedule(family=method, budget=2, alpha=1.0),
                        state_lr=0.1, rho=1.0, phi=model.phi)
    assert all(g.is_cuda and torch.isfinite(g).all() for g in grads)
