# Copyright (c) Phigent Robotics. All rights reserved.
# align_after_view_transfromation=True
# mAP: 0.3605
# mATE: 0.6189
# mASE: 0.2739
# mAOE: 0.4818
# mAVE: 0.3959
# mAAE: 0.2037
# NDS: 0.4828
# Eval time: 142.2s
#
# Per-class results:
# Object Class	AP	ATE	ASE	AOE	AVE	AAE
# car	0.577	0.446	0.154	0.079	0.324	0.190
# truck	0.280	0.647	0.217	0.094	0.308	0.205
# bus	0.375	0.701	0.208	0.078	0.894	0.303
# trailer	0.172	0.952	0.244	0.392	0.332	0.169
# construction_vehicle	0.090	0.754	0.424	1.018	0.107	0.341
# pedestrian	0.413	0.676	0.305	0.812	0.490	0.234
# motorcycle	0.315	0.668	0.266	0.676	0.514	0.178
# bicycle	0.273	0.534	0.281	1.066	0.198	0.010
# traffic_cone	0.564	0.415	0.334	nan	nan	nan
# barrier	0.546	0.395	0.304	0.123	nan	nan

# align_after_view_transfromation=False
# mAP: 0.3618
# mATE: 0.6168
# mASE: 0.2735
# mAOE: 0.4802
# mAVE: 0.3932
# mAAE: 0.2033
# NDS: 0.4842
# Eval time: 142.2s
#
# Per-class results:
# Object Class	AP	ATE	ASE	AOE	AVE	AAE
# car	0.577	0.445	0.154	0.079	0.320	0.190
# truck	0.280	0.648	0.217	0.094	0.305	0.205
# bus	0.373	0.700	0.208	0.082	0.896	0.304
# trailer	0.172	0.951	0.243	0.390	0.329	0.169
# construction_vehicle	0.091	0.750	0.422	1.008	0.109	0.339
# pedestrian	0.416	0.672	0.305	0.804	0.483	0.231
# motorcycle	0.318	0.665	0.267	0.674	0.506	0.179
# bicycle	0.274	0.533	0.281	1.070	0.199	0.010
# traffic_cone	0.567	0.411	0.334	nan	nan	nan
# barrier	0.550	0.391	0.304	0.122	nan	nan

num_gpu = 8
batch_size_per_device = 6
lr = 2e-4
weight_decay = 1e-7 # 1e-2 in bevdet and BEVFormerV2
lr_decay_steps = [16, 22] # bevdepth with epoch 24: [19, 23] # [16, 22] in bevbet
warmup_iters = 500 # 500 in bevdet
warmup_ratio = 0.001
max_epochs = 24


_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 448),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80

multi_adj_frame_id_cfg = (1, 1+1, 1)

model = dict(
    type='BEVDepth4D',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        pretrained='torchvision://resnet34',
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[256, 512],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[2,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=10, class_names=['car', 'truck',
                                            'construction_vehicle',
                                            'bus', 'trailer',
                                            'barrier',
                                            'motorcycle', 'bicycle',
                                            'pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=6.),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=1.5),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=500,

            # Scale-NMS
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]]
        )
    )
)

# Data
dataset_type = 'NuScenesDataset'
data_root = '/datasets/nuscenes/' # AI datacenter
ann_root = '/home/dlwpgh1994/3D-perception/data/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
                                'gt_depth'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=ann_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=batch_size_per_device,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
        data_root=data_root,
        ann_file=ann_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR')),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
data['train']['dataset'].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=lr, weight_decay=weight_decay)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=warmup_ratio,
    step=lr_decay_steps)
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SequentialControlHook',
        temporal_start_epoch=2,
    ),
]

# fp16 = dict(loss_scale='dynamic')