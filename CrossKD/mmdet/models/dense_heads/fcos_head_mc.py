# -*- coding: utf-8 -*-

from typing import Tuple, List
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmcv.cnn import Scale
from ..utils import multi_apply
from .fcos_head import FCOSHead


@MODELS.register_module()
class FCOSTeacherHeadMC(FCOSHead):
    """Adds Dropout2d(p) to the classification score map."""

    def __init__(self, *args, cls_dropout_p: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_dropout = nn.Dropout2d(cls_dropout_p)

    def forward_single(self, x: Tensor, scale: Scale, stride: int
                       ) -> Tuple[Tensor, Tensor, Tensor]:
        cls_score, bbox_pred, centerness = super().forward_single(
            x, scale, stride)

        # apply dropout on the score map (logits)
        cls_score = self.cls_dropout(cls_score)
        return cls_score, bbox_pred, centerness
    
    
    
    