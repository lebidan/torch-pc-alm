from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig
from .data import load_dataset
from .inference import Schedule, method_grad
from .metrics import mse_ce_accuracy, tree_cos
from .model import ResidualMLP
from .optim import make_adam


def train_one(config: ExperimentConfig, *, data_dir: str | Path = "data") -> dict[str, Any]:
    model_cfg = config.model
    method = config.method
    training = config.training
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(f"unsupported device: {config.device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    learning_rate = adam_learning_rate(model_cfg.width, model_cfg.depth, training.eta0, training.gamma0, training.learning_rate)
    x_train, y_train, x_test, y_test = load_dataset(
        config.dataset, train_subset=training.train_subset, test_subset=training.test_subset,
        seed=training.seed, data_dir=data_dir, input_dim=model_cfg.input_dim, output_dim=model_cfg.output_dim,
    )
    network = ResidualMLP(seed=training.seed, depth=model_cfg.depth, width=model_cfg.width,
                          input_dim=model_cfg.input_dim, output_dim=model_cfg.output_dim,
                          activation=model_cfg.activation, device=device)
    params = network.weights
    optimizer = make_adam(params, learning_rate)
    schedule = Schedule(family=method.name, budget=method.budget, alpha=method.alpha,
                        inner_steps=method.inner_steps, weight_credit_timing=method.weight_credit_timing)
    rows = []
    step = 0
    for epoch in range(training.epochs):
        for batch_idx in batch_order(x_train.shape[0], training.batch_size, training.seed + epoch, training.drop_last):
            xb = _tensor(x_train[batch_idx], device)
            yb = _tensor(y_train[batch_idx], device)
            optimizer.zero_grad(set_to_none=True)
            grads = method_grad(params, network.scales, network.skips, xb, yb, schedule,
                                state_lr=method.state_lr, rho=method.rho, phi=network.phi)
            for param, grad in zip(params, grads):
                param.grad = grad
            optimizer.step()
            step += 1
        train_mse, train_ce, train_acc = evaluate(network, x_train, y_train, training.batch_size, device)
        test_mse, test_ce, test_acc = evaluate(network, x_test, y_test, training.batch_size, device)
        rows.append({"epoch": epoch + 1, "step": step, "train_mse": train_mse,
                     "train_ce": train_ce, "train_acc": train_acc, "test_mse": test_mse,
                     "test_ce": test_ce, "test_acc": test_acc})

    diag_n = min(training.batch_size, x_train.shape[0])
    diag_x = _tensor(x_train[:diag_n], device)
    diag_y = _tensor(y_train[:diag_n], device)
    bp_grads = method_grad(params, network.scales, network.skips, diag_x, diag_y,
                           Schedule("bp", 0), state_lr=method.state_lr, rho=method.rho, phi=network.phi)
    method_grads = method_grad(params, network.scales, network.skips, diag_x, diag_y,
                               schedule, state_lr=method.state_lr, rho=method.rho, phi=network.phi)
    grad_cos_to_bp = float(tree_cos(method_grads, bp_grads).detach().cpu())
    final = {
        "dataset": config.dataset, "method": method.name, "width": model_cfg.width,
        "depth": model_cfg.depth, "activation": model_cfg.activation, "seed": training.seed,
        "budget": method.budget if method.name != "bp" else 0,
        "alpha": method.alpha if method.name == "pcalm" else 0.0,
        "state_lr": method.state_lr, "rho": method.rho, "learning_rate": learning_rate,
        "eta0": training.eta0, "gamma0": training.gamma0, "epochs": training.epochs,
        "batch_size": training.batch_size, "train_subset": training.train_subset,
        "test_subset": training.test_subset, "steps": step,
        "final_train_acc": rows[-1]["train_acc"], "final_test_acc": rows[-1]["test_acc"],
        "final_train_mse": rows[-1]["train_mse"], "final_test_mse": rows[-1]["test_mse"],
        "grad_cos_to_bp": grad_cos_to_bp,
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "metrics.csv")
    write_json(final, output_dir / "summary.json")
    write_json(asdict(config), output_dir / "config.json")
    return final


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device=device)


def adam_learning_rate(width: int, depth: int, eta0: float, gamma0: float, explicit_lr: float | None) -> float:
    if gamma0 != 1.0:
        raise ValueError("This reference implementation requires gamma0=1 (fixed model parameterization).")
    if explicit_lr is not None:
        return float(explicit_lr)
    return float(eta0 * (gamma0**2) * math.sqrt(width / depth))


def evaluate(network: ResidualMLP, X: np.ndarray, Y: np.ndarray, batch_size: int, device: torch.device) -> tuple[float, float, float]:
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    with torch.inference_mode():
        for start in range(0, X.shape[0], batch_size):
            stop = min(start + batch_size, X.shape[0])
            x = _tensor(X[start:stop], device)
            y = _tensor(Y[start:stop], device)
            metrics = mse_ce_accuracy(network(x), y)
            n = stop - start
            totals += np.array([float(v.cpu()) for v in metrics]) * n
            count += n
    return tuple((totals / max(count, 1)).tolist())


def batch_order(n: int, batch_size: int, seed: int, drop_last: bool) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    usable = (n // batch_size) * batch_size if drop_last else n
    perm = perm[:usable]
    return [perm[start : start + batch_size] for start in range(0, usable, batch_size)]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(value: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
