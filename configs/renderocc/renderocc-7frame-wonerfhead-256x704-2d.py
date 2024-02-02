_base_ = ['./bevstereo-occ-256x704-2d.py']

model = dict(
    type='RenderOcc2D',
    balance_cls_weight=True,
    final_softplus=True,
    use_3d_loss=True,
    nerf_head=None
)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)

depth_gt_path = 'data/nuscenes/depth_gt'
semantic_gt_path = 'data/nuscenes/seg_gt_lidarseg'

data = dict(
    samples_per_gpu=4,  # with 8 GPU, Batch Size=16
    workers_per_gpu=4,
    train=dict(
        use_rays=True,
        depth_gt_path=depth_gt_path,
        semantic_gt_path=semantic_gt_path,
        aux_frames=[-3,-2,-1,1,2,3],
        max_ray_nums=38400,
    )
)


runner = dict(type='EpochBasedRunner', max_epochs=24)

log_config = dict(
    interval=50,
)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

# 1 epoch
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 2.27
# ===> barrier - IoU = 39.86
# ===> bicycle - IoU = 8.64
# ===> bus - IoU = 29.36
# ===> car - IoU = 43.1
# ===> construction_vehicle - IoU = 18.52
# ===> motorcycle - IoU = 18.74
# ===> pedestrian - IoU = 20.12
# ===> traffic_cone - IoU = 15.69
# ===> trailer - IoU = 20.46
# ===> truck - IoU = 26.73
# ===> driveable_surface - IoU = 77.58
# ===> other_flat - IoU = 37.03
# ===> sidewalk - IoU = 45.31
# ===> terrain - IoU = 49.69
# ===> manmade - IoU = 37.92
# ===> vegetation - IoU = 29.95
# ===> mIoU of 6019 samples: 30.65
# {'mIoU': array([0.02265853, 0.39862453, 0.08640012, 0.29363337, 0.43095208,
#        0.18517379, 0.18740835, 0.20115947, 0.15688547, 0.20461213,
#        0.26733877, 0.77578176, 0.37033905, 0.45310229, 0.49688973,
#        0.37924946, 0.29952106, 0.89526128])}

# 4 epoch
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 7.35
# ===> barrier - IoU = 44.19
# ===> bicycle - IoU = 17.91
# ===> bus - IoU = 37.52
# ===> car - IoU = 47.5
# ===> construction_vehicle - IoU = 21.71
# ===> motorcycle - IoU = 20.32
# ===> pedestrian - IoU = 21.65
# ===> traffic_cone - IoU = 23.42
# ===> trailer - IoU = 27.05
# ===> truck - IoU = 33.96
# ===> driveable_surface - IoU = 80.38
# ===> other_flat - IoU = 40.82
# ===> sidewalk - IoU = 50.03
# ===> terrain - IoU = 53.94
# ===> manmade - IoU = 43.34
# ===> vegetation - IoU = 35.91
# ===> mIoU of 6019 samples: 35.71
# {'mIoU': array([0.07348384, 0.44191701, 0.17911529, 0.37516472, 0.47502992,
#        0.21713266, 0.20324715, 0.2164941 , 0.2341757 , 0.27050008,
#        0.33962312, 0.80383634, 0.40816429, 0.50028192, 0.53944564,
#        0.4334317 , 0.35909782, 0.90395509])}

# 12 epoch
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 10.13
# ===> barrier - IoU = 46.62
# ===> bicycle - IoU = 21.32
# ===> bus - IoU = 40.54
# ===> car - IoU = 49.77
# ===> construction_vehicle - IoU = 23.57
# ===> motorcycle - IoU = 22.36
# ===> pedestrian - IoU = 23.23
# ===> traffic_cone - IoU = 25.06
# ===> trailer - IoU = 30.07
# ===> truck - IoU = 36.07
# ===> driveable_surface - IoU = 81.62
# ===> other_flat - IoU = 41.72
# ===> sidewalk - IoU = 52.1
# ===> terrain - IoU = 55.45
# ===> manmade - IoU = 45.7
# ===> vegetation - IoU = 38.42
# ===> mIoU of 6019 samples: 37.87
# {'mIoU': array([0.10127372, 0.46618447, 0.2131633 , 0.40537036, 0.49773439,
#        0.23571383, 0.22358835, 0.2323348 , 0.25062111, 0.300734  ,
#        0.36074283, 0.8162047 , 0.41715579, 0.52104534, 0.55446332,
#        0.45704727, 0.38424077, 0.9047661 ])}

# 24 epoch
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 10.78
# ===> barrier - IoU = 47.47
# ===> bicycle - IoU = 21.92
# ===> bus - IoU = 41.59
# ===> car - IoU = 50.48
# ===> construction_vehicle - IoU = 22.67
# ===> motorcycle - IoU = 23.35
# ===> pedestrian - IoU = 23.75
# ===> traffic_cone - IoU = 25.65
# ===> trailer - IoU = 30.47
# ===> truck - IoU = 37.02
# ===> driveable_surface - IoU = 82.12
# ===> other_flat - IoU = 41.59
# ===> sidewalk - IoU = 53.13
# ===> terrain - IoU = 55.96
# ===> manmade - IoU = 46.61
# ===> vegetation - IoU = 38.99
# ===> mIoU of 6019 samples: 38.44
# {'mIoU': array([0.10784429, 0.47465584, 0.21917365, 0.41591472, 0.50484126,
#        0.2267084 , 0.23348093, 0.23746248, 0.25645269, 0.30472868,
#        0.37017564, 0.82120529, 0.41594669, 0.53134577, 0.55958817,
#        0.46609455, 0.38991094, 0.90224119])}