import mmcv
import torch

from mmdet.models.dense_heads import ALSAHead


def test_alsa_head_loss():
   
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    train_cfg = mmcv.Config(
        dict(
            assigner=dict(type='alsaAssigner', topk=17),
            allowed_border=-1,
            pos_weight=-1,
            debug=False))
    self = ALSAHead(
        num_classes=4,
        in_channels=1,
        train_cfg=train_cfg,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0))
    feat = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    cls_scores, bbox_preds, centernesses = self.forward(feat)

   
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, centernesses,
                                gt_bboxes, gt_labels, img_metas,
                                gt_bboxes_ignore)
    
    empty_cls_loss = sum(empty_gt_losses['loss_cls'])
    empty_box_loss = sum(empty_gt_losses['loss_bbox'])
    empty_centerness_loss = sum(empty_gt_losses['loss_centerness'])
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')
    assert empty_centerness_loss.item() == 0, (
        'there should be no centerness loss when there are no true boxes')

    
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, centernesses, gt_bboxes,
                              gt_labels, img_metas, gt_bboxes_ignore)
    onegt_cls_loss = sum(one_gt_losses['loss_cls'])
    onegt_box_loss = sum(one_gt_losses['loss_bbox'])
    onegt_centerness_loss = sum(one_gt_losses['loss_centerness'])
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'
    assert onegt_centerness_loss.item() > 0, (
        'centerness loss should be non-zero')
