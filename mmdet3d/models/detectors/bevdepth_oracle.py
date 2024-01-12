# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from .bevdet import BEVDet, BEVDet4D
from .bevdepth import BEVDepth

@DETECTORS.register_module()
class BEVDepthOracle(CenterPoint):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self, img_backbones, img_necks, 
                 img_view_transformers, 
                 img_bev_encoder_backbone,
                 img_bev_encoder_neck, **kwargs):
        super(CenterPoint, self).__init__(**kwargs)
        # Multi image backbone
        self.img_backbones = []
        for img_backbone in img_backbones:
            self.img_backbones.append(builder.build_backbone(img_backbone))
            
        # Multi image neck
        self.img_necks = []
        for img_neck in img_necks:
            self.img_necks.append(builder.build_neck(img_neck))
            
        # Multi view transformer
        self.img_view_transformers = []
        for img_view_transformer in img_view_transformers:
            self.img_view_transformers.append(builder.build_neck(img_view_transformer))
            
        # Global BEV encoder
        self.img_bev_encoder_backbone = \
            builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

        self.img_resolutions = ['256x448', '400x704', '544x960', '720x1280']

    def image_encoder(self, branch_idx, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        # print(next(self.img_backbones[branch_idx].parameters()).device)
        x = self.img_backbones[branch_idx](imgs)
        x = self.img_necks[branch_idx](x)
        if type(x) in [list, tuple]:
            x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def prepare_inputs(self, 
                       inputs, 
                       roi_cam_indexes: torch.Tensor = None):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs
            
        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()
        
        # use only roi inputs
        if roi_cam_indexes is not None:
            imgs = torch.index_select(imgs, 1, roi_cam_indexes)
            sensor2keyegos = torch.index_select(sensor2keyegos, 1, roi_cam_indexes)
            ego2globals = torch.index_select(ego2globals, 1, roi_cam_indexes)
            intrins = torch.index_select(intrins, 1, roi_cam_indexes)
            
            post_rots = torch.index_select(post_rots, 1, roi_cam_indexes)
            post_trans = torch.index_select(post_trans, 1, roi_cam_indexes)

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]
        
        
    
    def extract_feat_bevdet(self, branch_idx, points, img, img_metas, roi_cam_indexes=None, **kwargs):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat_bevdet(branch_idx, img, img_metas, roi_cam_indexes=None, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth)  
    
    def extract_feat_bevdepth(self, branch_idx, points, img, img_metas, roi_cam_indexes=None, **kwargs):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat_bevdepth(branch_idx, img, img_metas, roi_cam_indexes=None, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth)  
    
    """BEVDet version of extract_img_feat"""
    def extract_img_feat_bevdet(self, branch_idx, img, img_metas, roi_cam_indexes=None, **kwargs):
        """Extract features of images."""
        img = self.prepare_inputs(img, roi_cam_indexes)
        x = self.image_encoder(branch_idx, img[0])
        x, depth = self.img_view_transformers[2*branch_idx+1]([x] + img[1:7])
        x = self.bev_encoder(x)
        return [x], depth
    
    """BEVDepth version of extract_img_feat"""
    def extract_img_feat_bevdepth(self, branch_idx, img, img_metas, roi_cam_indexes=None, **kwargs):
        """Extract features of images."""
        imgs, rots, trans, intrins, post_rots, post_trans, bda = img
        
        # Prepare mlp input
        # mlp_input = self.img_view_transformer.get_mlp_input(rots[0], trans[0], intrins, post_rots, post_trans, bda)
        mlp_input = self.img_view_transformers[2*branch_idx].get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        
        # Extract BEV feature
        x = self.image_encoder(branch_idx, imgs)
        bev_feat, depth = self.img_view_transformers[2*branch_idx]([x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input])
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

        for branch_idx, img_resolution in enumerate(self.img_resolutions):
            # BEVDepth forward
            img_input = img_inputs[img_resolution]
            img_feats, pts_feats, depth = self.extract_feat_bevdepth( # TODO
                branch_idx, points, img=img_input, img_metas=img_metas, **kwargs)
            
            # Get depth loss (only for training)
            gt_depth = kwargs['gt_depth'][img_resolution]
            
            loss_depth = self.img_view_transformers[2*branch_idx].get_depth_loss(gt_depth, depth) # BEVDepth
            losses = dict(loss_depth=loss_depth)
            
            # Get bbox prediction loss -> Head prediction update
            losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
            
            
            # BEVDet forward
            img_feats, pts_feats, _ = self.extract_feat_bevdet( # TODO
                branch_idx, points, img=img_input, img_metas=img_metas, **kwargs)
            losses = dict()
            losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
            
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     roi_cam_indexes=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    roi_cam_indexes=None, **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)
        
    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    roi_cam_indexes=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, 
            roi_cam_indexes=roi_cam_indexes, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs