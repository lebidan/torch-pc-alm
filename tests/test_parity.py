from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pcalm.config import MethodConfig
from pcalm.inference import energy, infer, method_loss
from pcalm.model import ResidualMLP

REFERENCE = json.loads((Path(__file__).with_name("jax_reference.json")).read_text())
TRAINING_REFERENCE = json.loads((Path(__file__).with_name("jax_training_reference.json")).read_text())
X = torch.tensor(REFERENCE["x"])
Y = torch.tensor(REFERENCE["y"])


def as_params(layers):
    """JAX per-layer weight list -> ResidualMLP's (w_in, stacked w_mid, w_out)."""
    layers = [torch.tensor(v) for v in layers]
    return [layers[0], torch.stack(layers[1:-1]), layers[-1]]


def reference_model(activation: str) -> ResidualMLP:
    w_in, w_mid, w_out = as_params(REFERENCE["params"])
    model = ResidualMLP(seed=0, depth=w_mid.shape[0] + 2, width=w_in.shape[0], input_dim=w_in.shape[1],
                        output_dim=w_out.shape[0], activation=activation)
    model.load_state_dict({"w_in": w_in, "w_mid": w_mid, "w_out": w_out})
    return model


def check(actual, expected, *, atol=2e-5):
    torch.testing.assert_close(actual.detach(), torch.as_tensor(expected), atol=atol, rtol=2e-5)


def check_grads(model, expected):
    for p, e in zip(model.parameters(), as_params(expected)):
        check(p.grad, e)


def weight_grads(model, method):
    model.zero_grad(set_to_none=True)
    method_loss(model, X, Y, method).backward()
    return model


@pytest.mark.parametrize("activation", ["linear", "tanh", "relu"])
def test_fixed_array_jax_parity(activation):
    case = REFERENCE["cases"][activation]
    model = reference_model(activation)
    z0 = model.feedforward_hidden(X).detach()
    check(z0, case["acts"][:-1])
    check(model(X), case["acts"][-1])
    check(model.constraint_residuals(X, z0), case["residuals0"])
    check(energy(model, X, Y, z0, torch.zeros_like(z0), 1.0), case["energy0"])

    pc = MethodConfig(name="pc", budget=3, state_lr=0.1, rho=1.0)
    check(infer(model, X, Y, MethodConfig(name="pc", budget=1, state_lr=0.1))[0], case["first_free"])
    check(infer(model, X, Y, pc)[0], case["pc_free"])
    check_grads(weight_grads(model, MethodConfig(name="bp")), case["bp_grad"])
    check_grads(weight_grads(model, pc), case["pc_grad"])
    for timing in ("pre_dual_energy", "post_dual_energy"):
        expected = case[timing]
        pcalm = MethodConfig(name="pcalm", budget=3, alpha=1.0, rho=1.0, state_lr=0.1,
                             inner_steps=2, weight_credit_timing=timing)
        z, lam = infer(model, X, Y, pcalm)
        check(z, expected["free"])
        check(lam, expected["dual"])
        check_grads(weight_grads(model, pcalm), expected["grad"])


def test_one_adam_update_matches_jax():
    model = weight_grads(reference_model("tanh"), MethodConfig(name="bp"))
    torch.optim.Adam(model.parameters(), lr=0.001).step()
    for p, e in zip(model.parameters(), as_params(REFERENCE["adam_tanh_bp"])):
        check(p, e)


@pytest.mark.parametrize("name", ["bp", "pc", "pcalm"])
def test_three_training_steps_match_jax(name):
    model = reference_model("tanh")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    method = MethodConfig(name=name, budget=0 if name == "bp" else 3, alpha=1.0, state_lr=0.1, rho=1.0)
    for indices, expected in zip(TRAINING_REFERENCE["batch_indices"], TRAINING_REFERENCE["methods"][name]):
        optimizer.zero_grad(set_to_none=True)
        method_loss(model, X[indices], Y[indices], method).backward()
        check_grads(model, expected["grads"])
        optimizer.step()
        for p, e in zip(model.parameters(), as_params(expected["params"])):
            check(p, e)
