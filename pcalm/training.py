from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig
from .data import load_dataset
from .inference import method_loss
from .metrics import grad_cosine, mse_ce_accuracy
from .model import ResidualMLP


def train_one(config: ExperimentConfig, *, data_dir: str | Path = "data") -> dict[str, Any]:
    model_cfg = config.model
    method = config.method
    training = config.training
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(f"unsupported device: {config.device}")
    # Compile afresh for each run: a grid runs many model shapes in one process,
    # and past the recompile limit torch.compile falls back to eager with only a warning.
    torch.compiler.reset()
    learning_rate = adam_learning_rate(model_cfg.width, model_cfg.depth, training.eta0, training.gamma0, training.learning_rate)
    x_train, y_train, x_test, y_test = load_dataset(
        config.dataset, train_subset=training.train_subset, test_subset=training.test_subset,
        seed=training.seed, data_dir=data_dir, input_dim=model_cfg.input_dim, output_dim=model_cfg.output_dim,
    )
    model = ResidualMLP(seed=training.seed, depth=model_cfg.depth, width=model_cfg.width,
                        input_dim=model_cfg.input_dim, output_dim=model_cfg.output_dim,
                        activation=model_cfg.activation, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rows = []
    step = 0
    for epoch in range(training.epochs):
        for batch_idx in batch_order(x_train.shape[0], training.batch_size, training.seed + epoch, training.drop_last):
            x = torch.as_tensor(x_train[batch_idx], device=device)
            y = torch.as_tensor(y_train[batch_idx], device=device)
            optimizer.zero_grad(set_to_none=True)
            method_loss(model, x, y, method).backward()
            optimizer.step()
            step += 1
        train_mse, train_ce, train_acc = evaluate(model, x_train, y_train, training.batch_size, device)
        test_mse, test_ce, test_acc = evaluate(model, x_test, y_test, training.batch_size, device)
        rows.append({"epoch": epoch + 1, "step": step, "train_mse": train_mse,
                     "train_ce": train_ce, "train_acc": train_acc, "test_mse": test_mse,
                     "test_ce": test_ce, "test_acc": test_acc})

    diag_n = min(training.batch_size, x_train.shape[0])
    diag_x = torch.as_tensor(x_train[:diag_n], device=device)
    diag_y = torch.as_tensor(y_train[:diag_n], device=device)
    params = list(model.parameters())
    bp_grads = torch.autograd.grad(method_loss(model, diag_x, diag_y, replace(method, name="bp")), params)
    method_grads = torch.autograd.grad(method_loss(model, diag_x, diag_y, method), params)
    grad_cos_to_bp = grad_cosine(method_grads, bp_grads).item()
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


def adam_learning_rate(width: int, depth: int, eta0: float, gamma0: float, explicit_lr: float | None) -> float:
    if gamma0 != 1.0:
        raise ValueError("This reference implementation requires gamma0=1 (fixed model parameterization).")
    if explicit_lr is not None:
        return float(explicit_lr)
    return float(eta0 * (gamma0**2) * math.sqrt(width / depth))


@torch.inference_mode()
def evaluate(model: ResidualMLP, X: np.ndarray, Y: np.ndarray, batch_size: int,
             device: torch.device) -> tuple[float, float, float]:
    # Sum on the device and read the result once, instead of waiting for the GPU after every batch.
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    for start in range(0, X.shape[0], batch_size):
        x = torch.as_tensor(X[start:start + batch_size], device=device)
        y = torch.as_tensor(Y[start:start + batch_size], device=device)
        totals += mse_ce_accuracy(model(x), y) * x.shape[0]
    return tuple((totals / X.shape[0]).tolist())


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
