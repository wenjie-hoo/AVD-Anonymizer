dataset_type = 'CocoDataset'
data_root = '../../data/pp4av_dataset/'
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
    batch_size=8,
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
        pipeline=test_pipeline,
        metainfo=dict(classes=['face', 'license_plate'], 
                      palette=[(255, 0, 0), (0, 255, 0)]) 
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/pp4av_dataset/annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True, 
    metric_items=['mAP', 'mAP_50', 'mAP_75'],
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/pp4av_dataset/annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True, 
    metric_items=['mAP', 'mAP_50', 'mAP_75'], 
)
