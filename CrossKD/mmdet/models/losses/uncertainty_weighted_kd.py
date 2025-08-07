from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


from mmdet.registry import MODELS

from .utils import weight_reduce_loss 

EPS = 1e-6


@MODELS.register_module()
class UncertaintyWeightedKDLoss(nn.Module):


    def __init__(
        self,
        kd_weight: float = 1.0,
        tau: float = 10.0,
        reduction: str = "mean",
        uncertainty_mode: str = "entropy",
    ) -> None:
        super().__init__()
        assert reduction in {"mean", "sum", "none"}
        assert uncertainty_mode in {"entropy", "variance"}
        self.kd_weight = kd_weight
        self.tau = float(tau)
        self.reduction = reduction
        self.uncertainty_mode = uncertainty_mode

    @staticmethod
    def _entropy_weight(p: torch.Tensor) -> torch.Tensor:
        """Return (1 – normalised entropy) in [0, 1]."""
        ent = -(p * (p + EPS).log()).sum(dim=-1, keepdim=True)
        ent = ent / torch.log(torch.tensor(p.size(-1), device=p.device))
        return 1.0 - ent.detach()

    @staticmethod
    def _inv_var_weight(var: torch.Tensor) -> torch.Tensor:
        """Inverse-variance weight = exp(−σ²)."""
        return torch.exp(-var).detach()
    def forward(
        self,
        pred: torch.Tensor,
        soft_label: torch.Tensor | tuple | list,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[str] = None,
        teacher_var: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reduction = reduction_override if reduction_override else self.reduction
        if isinstance(soft_label, (tuple, list)):
            if len(soft_label) == 1:
                soft_label = soft_label[0]
            elif len(soft_label) == 2 and teacher_var is None:
                soft_label, teacher_var = soft_label
            else:
                raise ValueError(
                    f"Unexpected soft_label structure: {type(soft_label)} "
                    f"with length {len(soft_label)}")

        assert torch.is_tensor(soft_label), "soft_label must be a Tensor"
        assert pred.shape == soft_label.shape, (
            f"Shape mismatch pred {pred.shape} vs teacher {soft_label.shape}")
        target = F.softmax(soft_label / self.tau, dim=-1).detach()
        loss = F.kl_div(
            F.log_softmax(pred / self.tau, dim=-1),
            target,
            reduction="none",
        ) * (self.tau ** 2)
        if self.uncertainty_mode == "entropy":
            w_unc = self._entropy_weight(target) 
        else: 
            if teacher_var is None:
                raise ValueError(
                    "teacher_var must be provided when uncertainty_mode='variance'.")
            w_unc = self._inv_var_weight(teacher_var)

        if w_unc.ndim < loss.ndim:
            w_unc = w_unc.expand_as(loss)
        loss = loss * w_unc
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss * self.kd_weight
