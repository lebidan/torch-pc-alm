"""Measure one-method training steps after validating correctness."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcalm.inference import Schedule, method_grad
from pcalm.model import ResidualMLP
from pcalm.optim import make_adam


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BP, PC, or PC-ALM training steps.")
    parser.add_argument("--method", choices=["bp", "pc", "pcalm"], default="pcalm")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available to PyTorch")
    device = torch.device(args.device)
    model = ResidualMLP(seed=0, depth=args.depth, width=args.width, input_dim=784,
                        output_dim=10, activation="relu", device=device)
    optimizer = make_adam(model.parameters(), 1e-3)
    schedule = Schedule(args.method, 0 if args.method == "bp" else args.budget, alpha=1.0)
    x = torch.randn(args.batch_size, 784, device=device)
    labels = torch.randint(0, 10, (args.batch_size,), device=device)
    y = torch.nn.functional.one_hot(labels, 10).float()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        grads = method_grad(model.weights, model.scales, model.skips, x, y, schedule,
                            state_lr=0.25, rho=1.0, phi=model.phi)
        for p, g in zip(model.weights, grads):
            p.grad = g
        optimizer.step()

    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print({"method": args.method, "width": args.width, "depth": args.depth,
           "budget": schedule.budget, "batch_size": args.batch_size,
           "warmup_steps": args.warmup, "measured_steps": args.steps,
           "seconds_per_step": elapsed / args.steps,
           "peak_allocated_mib": (torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None)})


if __name__ == "__main__":
    main()
