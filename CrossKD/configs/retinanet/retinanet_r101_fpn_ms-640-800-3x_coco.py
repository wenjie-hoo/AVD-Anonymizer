_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
# optimizer



dataset_type = 'CocoDataset'
data_root = 'data/pp4av_dataset/'
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root='data/pp4av_dataset/',
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=None,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ],
        metainfo=dict(classes=['face', 'license_plate'])))
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type='CocoDataset',
#         data_root='data/pp4av_dataset/',
#         ann_file='annotations/instances_val.json',
#         data_prefix=dict(img='val/'),
#         test_mode=True,
#         pipeline=[
#             dict(
#                 type='LoadImageFromFile',
#                 file_client_args=dict(backend='disk')),
#             dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(
#                 type='PackDetInputs',
#                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                            'scale_factor'))
#         ]))
# test_dataloader = dict(
#     batch_size=16,
#     num_workers=4,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type='CocoDataset',
#         data_root='data/pp4av_dataset/',
#         ann_file='annotations/instances_val.json',
#         data_prefix=dict(img='val/'),
#         test_mode=True,
#         pipeline=[
#             dict(
#                 type='LoadImageFromFile',
#                 file_client_args=dict(backend='disk')),
#             dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(
#                 type='PackDetInputs',
#                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                            'scale_factor'))
#         ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/pp4av_dataset/',
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline,  # Use the defined pipeline variable
        metainfo=dict(classes=['face', 'license_plate'], 
                      palette=[(255, 0, 0), (0, 255, 0)])  # Add palette for visualization
    )
)

test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/pp4av_dataset/',
        ann_file='annotations/instances_val.json', 
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline,  # Use the defined pipeline variable
        metainfo=dict(classes=['face', 'license_plate'], 
                      palette=[(255, 0, 0), (0, 255, 0)])  # Add palette for visualization
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/pp4av_dataset/annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,  # Add class-wise evaluation
    metric_items=['mAP', 'mAP_50', 'mAP_75'],  # Specify metrics
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/pp4av_dataset/annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,  # Add class-wise evaluation
    metric_items=['mAP', 'mAP_50', 'mAP_75'],  # Specify metrics
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
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


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=35, norm_type=2))
auto_scale_lr = dict(enable=False, base_batch_size=16)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
