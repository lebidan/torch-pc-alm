from __future__ import annotations

import json
from pathlib import Path

import torch

from pcalm.inference import Schedule, al_energy_shifted, constraint_residuals, method_grad, run_pc, run_pcalm
from pcalm.model import activation_fn, forward
from pcalm.optim import make_adam

REFERENCE = json.loads((Path(__file__).with_name("jax_reference.json")).read_text())
TRAINING_REFERENCE = json.loads((Path(__file__).with_name("jax_training_reference.json")).read_text())


def tensors(values):
    return [torch.tensor(v, dtype=torch.float32) for v in values]


def check(actual, expected, *, atol=2e-5):
    for a, e in zip(actual, tensors(expected)):
        torch.testing.assert_close(a.detach(), e, atol=atol, rtol=2e-5)


def test_fixed_array_jax_parity():
    x = torch.tensor(REFERENCE["x"], dtype=torch.float32)
    y = torch.tensor(REFERENCE["y"], dtype=torch.float32)
    scales = REFERENCE["scales"]
    skips = tuple(REFERENCE["skips"])
    for activation, case in REFERENCE["cases"].items():
        params = [torch.nn.Parameter(v) for v in tensors(REFERENCE["params"])]
        phi = activation_fn(activation)
        acts = forward(params, scales, skips, x, phi)
        check(acts, case["acts"])
        free0 = [z.detach() for z in acts[:-1]]
        duals0 = [torch.zeros_like(z) for z in free0]
        check(constraint_residuals(params, scales, skips, x, free0, phi), case["residuals0"])
        torch.testing.assert_close(al_energy_shifted(params, scales, skips, x, y, free0, duals0, 1.0, phi),
                                   torch.tensor(case["energy0"]), atol=2e-5, rtol=2e-5)
        first_free, _ = run_pc(params, scales, skips, x, y, state_lr=0.1, rho=1.0, steps=1, phi=phi)
        check(first_free, case["first_free"])
        pc_free, _ = run_pc(params, scales, skips, x, y, state_lr=0.1, rho=1.0, steps=3, phi=phi)
        check(pc_free, case["pc_free"])
        check(method_grad(params, scales, skips, x, y, Schedule("bp", 0), state_lr=0.1, rho=1.0, phi=phi), case["bp_grad"])
        check(method_grad(params, scales, skips, x, y, Schedule("pc", 3), state_lr=0.1, rho=1.0, phi=phi), case["pc_grad"])
        for timing in ("pre_dual_energy", "post_dual_energy"):
            expected = case[timing]
            free, duals = run_pcalm(params, scales, skips, x, y, state_lr=0.1, rho=1.0, alpha=1.0,
                                    budget=3, inner_steps=2, weight_credit_timing=timing, phi=phi)
            check(free, expected["free"])
            check(duals, expected["dual"])
            check(method_grad(params, scales, skips, x, y, Schedule("pcalm", 3, 1.0, 2, timing),
                              state_lr=0.1, rho=1.0, phi=phi), expected["grad"])


def test_one_adam_update_matches_jax():
    params = [torch.nn.Parameter(v) for v in tensors(REFERENCE["params"])]
    x = torch.tensor(REFERENCE["x"], dtype=torch.float32)
    y = torch.tensor(REFERENCE["y"], dtype=torch.float32)
    grads = method_grad(params, REFERENCE["scales"], tuple(REFERENCE["skips"]), x, y,
                        Schedule("bp", 0), state_lr=0.1, rho=1.0, phi=activation_fn("tanh"))
    optimizer = make_adam(params, 0.001)
    for p, g in zip(params, grads):
        p.grad = g
    optimizer.step()
    check(params, REFERENCE["adam_tanh_bp"])


def test_three_training_steps_match_jax():
    x = torch.tensor(REFERENCE["x"], dtype=torch.float32)
    y = torch.tensor(REFERENCE["y"], dtype=torch.float32)
    for method, history in TRAINING_REFERENCE["methods"].items():
        params = [torch.nn.Parameter(v) for v in tensors(REFERENCE["params"])]
        optimizer = make_adam(params, 0.001)
        schedule = Schedule(family=method, budget=0 if method == "bp" else 3,
                            alpha=1.0, inner_steps=1, weight_credit_timing="pre_dual_energy")
        for indices, expected in zip(TRAINING_REFERENCE["batch_indices"], history):
            grads = method_grad(params, REFERENCE["scales"], tuple(REFERENCE["skips"]),
                                x[indices], y[indices], schedule, state_lr=0.1,
                                rho=1.0, phi=activation_fn("tanh"))
            check(grads, expected["grads"])
            optimizer.zero_grad(set_to_none=True)
            for param, grad in zip(params, grads):
                param.grad = grad
            optimizer.step()
            check(params, expected["params"])
