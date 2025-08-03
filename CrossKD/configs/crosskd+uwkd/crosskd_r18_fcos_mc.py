_base_ = [
    '../_base_/datasets/pp4av_dataset.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/default_runtime.py'
]

data_preprocessor = dict(type='DetDataPreprocessor',
                         mean=[102.9801, 115.9465, 122.7717],
                         std=[1.0, 1.0, 1.0],
                         bgr_to_rgb=False,
                         pad_size_divisor=32)

# teacher_ckpt = 'work_dirs/fcos_r50-caffe_fpn_gn-head_2x_coco/epoch_24.pth'
teacher_ckpt = 'work_dirs/fcos_r101-caffe_fpn_gn-head-1x_coco/exp_1/epoch_24.pth'

model = dict(
    type='CrossKDFCOS',
    data_preprocessor=data_preprocessor,
#     teacher_config='configs/fcos/fcos_r50-caffe_fpn_gn-head_2x_coco.py',
    teacher_config='configs/fcos/fcos_r101-caffe_fpn_gn-head-1x_coco.py',
    teacher_ckpt=teacher_ckpt,
    backbone=dict(
        type='ResNet',
        depth=18,  # Keep ResNet-18 for speed
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],  # ResNet-18 channels
        out_channels=256,  # Match teacher exactly
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=2,
        in_channels=256, 
        stacked_convs=4, 
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    kd_cfg=dict(
#         loss_cls_kd=dict(type='KDQualityFocalLoss', beta=1, loss_weight=0.4),
        loss_cls_kd=dict(
            type='UncertaintyWeightedKDLoss',
            kd_weight=1.0,
            tau=10,
            reduction='mean',
#             uncertainty_mode='variance'), 
            uncertainty_mode='entropy'), 
        loss_reg_kd=dict(type='GIoULoss', loss_weight=1.0),
        reused_teacher_head_idx=2),  # Now safe to use
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=1,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# Dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 800)], 
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline),
    batch_size=4,
    num_workers=2)

# Optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.015), 
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

max_epochs = 24  
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2))
auto_scale_lr = dict(enable=True, base_batch_size=16)

# Learning rate
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]