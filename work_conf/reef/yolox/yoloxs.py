__CFGDIR__ = "../../../configs"
_base_ = [f'{__CFGDIR__}/_base_/schedules/schedule_1x.py', f'{__CFGDIR__}/_base_/default_runtime.py']

#mg_scale = (640, 640)
#img_scale = (960, 960)
img_scale = (736, 1280)
num_class = 1
# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    #random_size_range=(15, 25),
    #random_size_range=(35, 45),
    #random_size_range=(45, 55),
    #random_size_interval=10,
    random_size_interval=0,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=num_class, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
data_root = '../reef-data/out/all/'

dataset_type = 'CocoDataset'

classes = ['cots']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 1),
        #scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),        
    # this MixUp is different from original yolox mixup original is copy paste
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.1, 1),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    #dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    
    dict(type='Normalize', **img_norm_cfg),

    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations/annotations_train_2.json',
        img_prefix=data_root + 'images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations/annotations_valid_2.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        classes=classes,
        type=dataset_type,
        #ann_file=data_root + 'out/annotations_test_2.json',
        ann_file=data_root + 'annotations/annotations_valid_2.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

#max_epochs = 300
#num_last_epochs = 15

max_epochs = 20
num_last_epochs = 3

resume_from = None
interval = 1

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    #warmup_iters=5,  # 5 epoch
    warmup_iters=1,
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]

work_dir = f'./work_dirs/reef/yolos'

load_from = f'{data_root}\out\weights\yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
#load_from = f'{work_dir}/epoch_3.pth'

checkpoint_config = dict(interval=interval)
"""
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox')

"""


evaluation = dict(
    #classwise=True, #only one class
    interval=1, 
    metric=['bbox'],
    jsonfile_prefix=f"{work_dir}/valid")

log_config = dict(interval=50)

# seems not support fp16
#fp16 = dict(loss_scale=512.0)