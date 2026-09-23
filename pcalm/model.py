from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn

Params = list[torch.Tensor] | nn.ParameterList


def linear_activation(x: torch.Tensor) -> torch.Tensor:
    return x


def activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "linear":
        return linear_activation
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return torch.relu
    raise ValueError(f"unknown activation: {name}")


def activation_derivative(phi, x: torch.Tensor) -> torch.Tensor:
    if phi is linear_activation:
        return torch.ones_like(x)
    if phi is torch.tanh:
        activated = torch.tanh(x)
        return 1.0 - activated * activated
    if phi is torch.relu:
        return (x > 0).to(x.dtype)
    raise ValueError("packed inference supports linear, tanh, and relu activations")


def model_scales(width: int, depth: int, input_dim: int) -> list[float]:
    if depth < 2:
        raise ValueError("depth must include at least one hidden layer and one output layer")
    return [1.0 / math.sqrt(input_dim)] + [1.0 / math.sqrt(width * depth)] * (depth - 2) + [1.0 / width]


def skip_mask(depth: int) -> tuple[bool, ...]:
    return tuple([False] + [True] * (depth - 2) + [False])


def init_params(
    seed: int,
    *,
    depth: int,
    width: int,
    input_dim: int,
    output_dim: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> nn.ParameterList:
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    layers = nn.ParameterList()
    for layer_ix in range(depth):
        in_dim = input_dim if layer_ix == 0 else width
        out_dim = output_dim if layer_ix == depth - 1 else width
        layers.append(nn.Parameter(torch.randn((out_dim, in_dim), generator=generator, device=device, dtype=dtype)))
    return layers


def block_pred(
    W: torch.Tensor,
    scale: float,
    skip: bool,
    z_prev: torch.Tensor,
    phi: Callable[[torch.Tensor], torch.Tensor],
    *,
    is_first: bool,
) -> torch.Tensor:
    inp = z_prev if is_first else phi(z_prev)
    pred = scale * (inp @ W.T)
    if skip:
        pred = pred + z_prev
    return pred


def forward(params: Params, scales: list[float], skips: tuple[bool, ...], x: torch.Tensor, phi) -> list[torch.Tensor]:
    acts = []
    z_prev = x
    for layer_ix, W in enumerate(params):
        z_prev = block_pred(W, scales[layer_ix], skips[layer_ix], z_prev, phi, is_first=(layer_ix == 0))
        acts.append(z_prev)
    return acts


def logits(params: Params, scales: list[float], skips: tuple[bool, ...], x: torch.Tensor, phi) -> torch.Tensor:
    return forward(params, scales, skips, x, phi)[-1]


class ResidualMLP(nn.Module):
    def __init__(self, *, seed: int, depth: int, width: int, input_dim: int, output_dim: int, activation: str, device: torch.device | str):
        super().__init__()
        self.weights = init_params(seed, depth=depth, width=width, input_dim=input_dim, output_dim=output_dim, device=device)
        self.scales = model_scales(width, depth, input_dim)
        self.skips = skip_mask(depth)
        self.phi = activation_fn(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return logits(self.weights, self.scales, self.skips, x, self.phi)
