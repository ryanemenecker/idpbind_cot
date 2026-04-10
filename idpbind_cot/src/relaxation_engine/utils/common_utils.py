from __future__ import annotations

import torch


def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    keepdim: bool = False,
) -> torch.Tensor:
    """L2 norm with a clamped floor to avoid NaN gradients at zero."""
    sq_norm = torch.sum(x * x, dim=dim, keepdim=keepdim)
    return torch.sqrt(torch.clamp(sq_norm, min=eps))


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Unit-normalize a vector with clamped denominator."""
    return x / safe_norm(x, dim=dim, keepdim=True, eps=eps)
