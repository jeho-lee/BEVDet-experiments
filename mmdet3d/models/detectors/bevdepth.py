# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from .bevdet import BEVDet



@DETECTORS.register_module()
class BEVDepth(BEVDet):
    
    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        imgs, rots, trans, intrins, post_rots, post_trans, bda = img
        
        # Prepare mlp input
        # mlp_input = self.img_view_transformer.get_mlp_input(rots[0], trans[0], intrins, post_rots, post_trans, bda)
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        
        # Extract BEV feature
        x = self.image_encoder(imgs)
        bev_feat, depth = self.img_view_transformer([x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input])
        bev_feat = self.bev_encoder(bev_feat)
        return [bev_feat], depth
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        
        # Get depth loss (only for training)
        gt_depth = kwargs['gt_depth']
        
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth) # BEVDet
        # loss_depth = self.img_view_transformer.get_depth_loss_solofusion(gt_depth, depth) # SOLOFusion
        
        losses = dict(loss_depth=loss_depth)
        
        # Get bbox prediction loss
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        
        return losses