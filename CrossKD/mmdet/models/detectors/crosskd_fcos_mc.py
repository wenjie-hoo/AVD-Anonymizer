from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList

from .crosskd_fcos import CrossKDFCOS
from ..utils import multi_apply


@MODELS.register_module()
class CrossKDFCOS_MC(CrossKDFCOS):
    def __init__(self, *args, mc_samples: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_samples = mc_samples 

    def _flatten_cls(self, cls_scores: List[Tensor]) -> Tensor:
        """Concat per‑level cls scores to (A, C)."""
        return torch.cat([
            s.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
            for s in cls_scores
        ])
def loss(self, batch_inputs: Tensor,
         batch_data_samples: SampleList) -> dict:
    """KD with one cached teacher feature pass."""
    # student forward  (backbone + FPN + head)
    feats_S         = self.extract_feat(batch_inputs)
    cls_S, _, _     = self.bbox_head(feats_S)          # plain FCOS head
    logits_S        = torch.cat([                     # (B, A, C)
        s.permute(0, 2, 3, 1).reshape(s.size(0), -1,
                                       self.bbox_head.cls_out_channels)
        for s in cls_S], dim=1).flatten(0, 1)          # -> (A_tot, C)

    with torch.no_grad():
        self.teacher.eval()
        feats_T = self.teacher.extract_feat(batch_inputs)     # 1× backbone

    mc_logits = []
    self.teacher.bbox_head.train()                  
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(self.mc_samples):          
            cls_T, _, _ = self.teacher.bbox_head(feats_T)
            flat = torch.cat([
                s.permute(0, 2, 3, 1).reshape(s.size(0), -1,
                                                self.bbox_head.cls_out_channels)
                for s in cls_T], dim=1)
            mc_logits.append(flat)
    self.teacher.bbox_head.eval()

    mc_logits = torch.stack(mc_logits, 0)            # (N, B, A, C)
    teacher_mu  = mc_logits.mean(0).flatten(0, 1)    # (A_tot, C)
    teacher_var = mc_logits.var (0, unbiased=False).flatten(0, 1)


    loss_cls_kd = self.loss_cls_kd(
        pred        = logits_S,
        soft_label  = teacher_mu,
        teacher_var = teacher_var)


    stu_losses = self.bbox_head.loss_by_feat(
        cls_S,                                      # student cls scores
        *self.bbox_head(feats_S)[1:],               # bbox, centerness
        batch_gt_instances = [d.gt_instances for d in batch_data_samples],
        batch_img_metas    = [d.metainfo     for d in batch_data_samples],
        batch_gt_instances_ignore = None)

    stu_losses['loss_cls_kd'] = loss_cls_kd
    return stu_losses