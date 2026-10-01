"""The surrogate network (role A): (damage raster, interface parameters) -> capacity.

A small convolutional encoder reads the raster of the interface window; its
code is concatenated with the normalized parameters and an MLP predicts the
capacity, scaled to the family's training peak. Smooth activations
throughout, so the output is differentiable in the parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn

ACTS = {"silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh, "relu": nn.ReLU}


class RasterEncoder(nn.Module):
    """Strided convolutions, then a flatten and a linear layer. The flatten
    keeps position: where the rot is relative to the interface matters."""

    def __init__(self, in_shape=(32, 96), channels=(16, 32, 64), out_dim=64, act="silu"):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv2d(c_in, c, kernel_size=3, stride=2, padding=1), ACTS[act]()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 1, *in_shape)).numel()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(n, out_dim), ACTS[act]())

    def forward(self, raster):            # (B, 1, H, W) -> (B, out_dim)
        return self.fc(self.conv(raster))


class CapacityNet(nn.Module):
    def __init__(self, n_params: int, in_shape=(32, 96), enc_channels=(16, 32, 64), enc_out=64,
                 hidden=(256, 256, 256), act="silu"):
        super().__init__()
        self.encoder = RasterEncoder(in_shape, enc_channels, enc_out, act)
        layers, d = [], enc_out + n_params
        for h in hidden:
            layers += [nn.Linear(d, h), ACTS[act]()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, raster, params):    # (B, 1, H, W), (B, n_params) -> (B,)
        return self.head(torch.cat([self.encoder(raster), params], dim=1)).squeeze(1)

    def encode(self, raster):
        return self.encoder(raster)

    def from_code(self, code, params):
        """Capacity from a precomputed raster code: cheap when the damage is
        fixed and only the parameters move, as in an optimization loop."""
        return self.head(torch.cat([code, params], dim=1)).squeeze(1)


def build(cfg, n_params: int) -> CapacityNet:
    """From the `model` section of the hydra config."""
    return CapacityNet(n_params, in_shape=tuple(cfg.in_shape), enc_channels=tuple(cfg.enc_channels),
                       enc_out=cfg.enc_out, hidden=tuple(cfg.hidden), act=cfg.act)
