num_classes = 1
# model settings
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            #nms_pre=2000,
            #max_per_img=2000,
            nms_pre=500,
            max_per_img=500,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            #nms_pre=1000,
            #max_per_img=1000,
            nms_pre=500,
            max_per_img=500,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))


dataset_type = 'CocoDataset'
data_root = '../reef-data/'
classes = ['cots']

albu_train_transforms = [
    dict(type='VerticalFlip', p=0.5),
    dict(type='RandomRotate90', p=0.5),
    # dict(type='ColorJitter', p=0.5)
    # dict(type='ShiftScaleRotate', shift_limit=0.0625, scale_limit=0.9, rotate_limit=30, interpolation=1, p=0.5)
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (1440, 2560)


albu_train_transforms = [
    #dict(
    #    type='HorizontalFlip',
    #    p=0.5),
    #dict(
    #    type='VerticalFlip',
    #    p=0.5),
    dict(
        type='RandomRotate90',
        p=1),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='HueSaturationValue',
                #hue_limit=0.05, 
                #sat_limit=0.7, 
                #val_limit=0.4,
                hue_shift_limit=5, 
                sat_shift_limit=70, 
                val_shift_limit=40,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.15, 
                contrast_limit=0.15,
                p=0.5),
            dict(
                type='ToGray',
                p=0.1)], p=0.8),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.1, 
                rotate_limit=3, 
                scale_limit=(0.5, 2.0), 
                #min_bbox=27, 
                #max_bbox=None, 
                border_mode=0, 
                value=(114, 114, 114),
                p=0.4),
            dict(
                type='Perspective',
                scale=0.1, 
                pad_mode=0, 
                pad_val=(114, 114, 114),
                p=0.1)
            ], 
    p=0.8),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='GaussianBlur',
                blur_limit=(3, 5),
                p=0.5),
            dict(
                type='MotionBlur',
                blur_limit=(3, 5),
                p=0.5),
            dict(
                type='MultiplicativeNoise',
                multiplier=(0.9, 1.1), 
                per_channel=False, 
                elementwise=True,
                p=0.5)
        ], 
    p=0.2)
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(2560, 1440), keep_ratio=True),    
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),  
    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_area=16, 
            min_visibility=0.2,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),

  
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2560, 1440),
        flip=False,
        flip_direction=["horizontal","vertical"],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
            classes=classes,
            type=dataset_type,
            ann_file=data_root + 'out/all/annotations/annotations_train_2.json',
            img_prefix=data_root + 'out/all/images/',
            pipeline=train_pipeline),
    val=dict(
            classes=classes,
            type=dataset_type,
            ann_file=data_root + 'out/all/annotations/annotations_valid_2.json',
            img_prefix=data_root + 'out/all/images/',
            pipeline=test_pipeline),
    test=dict(
            classes=classes,
            type=dataset_type,
            ann_file=data_root + 'out/all/annotations/annotations_valid_2.json',
            img_prefix=data_root + 'out/all/images/',
            pipeline=test_pipeline)
)

nx = 1
work_dir = f'./work_dirs/reef/cascade-r50-large'
evaluation = dict(
    classwise=True,
    interval=1,
    metric=['bbox'],
    jsonfile_prefix=f"{work_dir}/valid")
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1/3,
    #step=[8 * nx, 11 * nx])
    step=[3 * nx])
custom_hooks = [dict(type='NumClassCheckHook')]
total_epochs = 5* nx
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

#checkpoint_config = dict(interval=total_epochs, save_optimizer=False)

checkpoint_config = dict(interval=1)

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'

#load_from = f'{data_root}/out/weights/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'
load_from = f'{data_root}/out/weights/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'

resume_from = None
workflow = [('train', 1)]
fp16 = dict(loss_scale=512.0)

