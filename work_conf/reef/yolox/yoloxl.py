_base_ = './yoloxs.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256))

    
data = dict(
    samples_per_gpu=4)

work_dir = work_dir = f'./work_dirs/reef/yoloxl'
data_root = '../reef-data/'
load_from = f'{data_root}\out\weights\yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'