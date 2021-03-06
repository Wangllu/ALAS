import torch
import torch.nn as nn
import numpy as np

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from mmdet.models.dense_heads.fcos_head import FCOSHead
from mmdet.models.dense_heads.gfl_head import GFLHead
from mmdet.models.dense_heads.autoassign_head import AutoAssignHead, EPS

@BBOX_ASSIGNERS.register_module()
class ALSAAssigner(BaseAssigner):
  

    def __init__(self,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

   
    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               x,
               loss_cls,
               gt_bboxes_ignore=None,
               gt_labels=None):
       
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

       
        # 计算所有边界框和gt之间的iou
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

       
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
           
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
     
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)


        # 计算所有bbox和gt之间的中心距离
        center_prior_list=[]
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        center_prior = torch.exp(-distances).prod(dim=-1)
        center_prior_list.append(center_prior)
        center_prior_weights = torch.cat(center_prior_list, dim=0)
        cls_score, bbox_pred, cls_feat, reg_feat = super(
            GFLHead, self).forward_single(x)
        confidence_weight = cls_score
        p_pos_weight = np.exp(-center_prior_weights-confidence_weight)
	    
        means_init = np.array([min_loss,max_loss]).reshape(2, 1)
        sigma_init = np.array([1.0]).reshape(2, 1, 1)
        Gauss = np.exp(-((p_pos_weight-means_init)**2)/(2*sigma_init**2)) / (sigma_init*np.sqrt(2*np.pi))
        optimizer = torch.optim.SGD([means_init,sigma_init],lr=2e-2)

        idx = list(range(len(p_pos_weight)))
        for epoch in range(2):
            np.random.shuffle(idx)
            for i in range(0,len(idx),10):
                x_batch = p_pos_weight[idx[i:i+10]]
        optimizer.zero_grad()
        Gauss = torch.distributions.Normal(loc=means_init, scale=sigma_init)
        negative_log_likelihood = -1 * torch.sum(Gauss.log_prob(x_batch))
        negative_log_likelihood.backward()
        optimizer.step()
        overlaps_mean_per_gt = means_init
        overlaps_std_per_gt = sigma_init
        overlaps_thr_per_gt = overlaps_mean_per_gt
        
        is_pos = p_pos_weight >= overlaps_thr_per_gt[None, :]
        candidate_idxs = []

 
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

      
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts

       
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
