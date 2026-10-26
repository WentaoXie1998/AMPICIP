# losses.py
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn


def _rbf_kernel(x: Tensor, sigma: float = None) -> Tensor:
    dist2 = torch.cdist(x, x, p=2.0) ** 2  # [B,B]

    if sigma is None:
        sigma = torch.median(dist2.detach())
        sigma = torch.sqrt(sigma + 1e-8)

    K = torch.exp(-dist2 / (2 * sigma ** 2 + 1e-8))
    return K


def hsic_loss(x: Tensor, y: Tensor) -> Tensor:
    assert x.shape[0] == y.shape[0], "batch size must match"
    B = x.shape[0]
    if B < 2:
        return x.new_tensor(0.0)

    Kx = _rbf_kernel(x)
    Ky = _rbf_kernel(y)

    I = torch.eye(B, device=x.device, dtype=x.dtype)
    H = I - 1.0 / B * torch.ones_like(I)

    Kc_x = H @ Kx @ H
    Kc_y = H @ Ky @ H

    hsic = torch.trace(Kc_x @ Kc_y) / ((B - 1) ** 2)
    return hsic


mse_loss_fn = nn.MSELoss()
