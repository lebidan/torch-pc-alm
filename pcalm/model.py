from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

ACTIVATIONS = {"linear": nn.Identity, "tanh": nn.Tanh, "relu": nn.ReLU}


class ResidualMLP(nn.Module):
    """Bias-free residual MLP with `depth - 1` hidden blocks and one output block.

    first block:   z_1 = s_in * x @ W_in.T                   (no activation on x)
    middle blocks: z_l = z_{l-1} + s_mid * phi(z_{l-1}) @ W_l.T
    output block:  out = s_out * phi(z_{L-1}) @ W_out.T      (no skip)

    The `depth - 2` middle weights share one shape, so they are stored stacked
    as a single `[depth - 2, width, width]` parameter. Hidden states `z` (the
    "free variables" `free` of the original JAX code) are stacked the same way,
    `[depth - 1, batch, width]`, so every middle-layer constraint is evaluated
    with a single batched matmul.
    """

    def __init__(self, *, seed: int, depth: int, width: int, input_dim: int, output_dim: int,
                 activation: str, device: torch.device | str = "cpu"):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must include at least one hidden layer and one output layer")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation: {activation}")
        # Draw on the CPU so a seed gives the same weights on every device.
        gen = torch.Generator().manual_seed(seed)
        self.w_in = nn.Parameter(torch.randn(width, input_dim, generator=gen))
        self.w_mid = nn.Parameter(torch.randn(depth - 2, width, width, generator=gen))
        self.w_out = nn.Parameter(torch.randn(output_dim, width, generator=gen))
        self.to(device)
        self.s_in = 1.0 / math.sqrt(input_dim)
        self.s_mid = 1.0 / math.sqrt(width * depth)
        self.s_out = 1.0 / width
        self.phi = ACTIVATIONS[activation]()

    def feedforward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward hidden states, stacked as `[depth - 1, batch, width]`."""
        z = self.s_in * F.linear(x, self.w_in)
        states = [z]
        for W in self.w_mid:
            z = z + self.s_mid * F.linear(self.phi(z), W)
            states.append(z)
        return torch.stack(states)

    def readout(self, z_last: torch.Tensor) -> torch.Tensor:
        return self.s_out * F.linear(self.phi(z_last), self.w_out)

    def constraint_residuals(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Hidden-edge constraints `c_l = z_l - block_l(z_{l-1})`; zero on the feed-forward pass."""
        first = z[0] - self.s_in * F.linear(x, self.w_in)
        middle = z[1:] - z[:-1] - self.s_mid * torch.bmm(self.phi(z[:-1]), self.w_mid.mT)
        return torch.cat((first[None], middle))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.feedforward_hidden(x)[-1])
