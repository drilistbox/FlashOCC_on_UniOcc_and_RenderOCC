_base_ = ['./bevstereo-occ-256x704-2d.py']

model = dict(
    type='RenderOcc2D',
    balance_cls_weight=True,
    final_softplus=True,
    use_3d_loss=True,
    nerf_head=dict(
        type='NerfHead',
        point_cloud_range=[-40, -40, -1, 40, 40, 5.4],
        voxel_size=0.4,
        scene_center=[0, 0, 2.2],
        radius=39,
        use_depth_sup=True,
        weight_depth=0.1,
        weight_semantic=0.1,
    )
    # nerf_head=None
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
# ===> others - IoU = 2.19
# ===> barrier - IoU = 40.04
# ===> bicycle - IoU = 7.51
# ===> bus - IoU = 31.28
# ===> car - IoU = 42.84
# ===> construction_vehicle - IoU = 17.37
# ===> motorcycle - IoU = 19.76
# ===> pedestrian - IoU = 18.44
# ===> traffic_cone - IoU = 16.71
# ===> trailer - IoU = 20.5
# ===> truck - IoU = 27.35
# ===> driveable_surface - IoU = 77.33
# ===> other_flat - IoU = 36.51
# ===> sidewalk - IoU = 45.32
# ===> terrain - IoU = 49.83
# ===> manmade - IoU = 37.9
# ===> vegetation - IoU = 29.98
# ===> mIoU of 6019 samples: 30.64
# {'mIoU': array([0.02185315, 0.40041932, 0.07505111, 0.31280221, 0.42844366,
#        0.17374394, 0.19760528, 0.18435559, 0.16706522, 0.20504521,
#        0.27349894, 0.77333741, 0.36509148, 0.45321694, 0.49826651,
#        0.37900015, 0.2997506 , 0.89549414])}

# 4 epoch
# ===> mIoU of 6019 samples: 35.96
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 6.69
# ===> barrier - IoU = 44.07
# ===> bicycle - IoU = 19.03
# ===> bus - IoU = 39.2
# ===> car - IoU = 47.52
# ===> construction_vehicle - IoU = 20.72
# ===> motorcycle - IoU = 21.46
# ===> pedestrian - IoU = 22.08
# ===> traffic_cone - IoU = 23.98
# ===> trailer - IoU = 27.06
# ===> truck - IoU = 33.9
# ===> driveable_surface - IoU = 80.47
# ===> other_flat - IoU = 40.76
# ===> sidewalk - IoU = 50.13
# ===> terrain - IoU = 54.16
# ===> manmade - IoU = 43.06
# ===> vegetation - IoU = 36.14
# ===> mIoU of 6019 samples: 35.91
# {'mIoU': array([0.06692937, 0.44068219, 0.19026138, 0.39196683, 0.47519653,
#        0.20717313, 0.21463704, 0.22075909, 0.23982669, 0.27056592,
#        0.33895972, 0.80467785, 0.40762568, 0.50127821, 0.54156186,
#        0.43061602, 0.36144933, 0.90389957])}


# 12 epoch
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 10.15
# ===> barrier - IoU = 46.35
# ===> bicycle - IoU = 21.22
# ===> bus - IoU = 41.22
# ===> car - IoU = 49.65
# ===> construction_vehicle - IoU = 23.1
# ===> motorcycle - IoU = 22.44
# ===> pedestrian - IoU = 23.41
# ===> traffic_cone - IoU = 25.59
# ===> trailer - IoU = 32.88
# ===> truck - IoU = 36.74
# ===> driveable_surface - IoU = 81.71
# ===> other_flat - IoU = 42.39
# ===> sidewalk - IoU = 52.46
# ===> terrain - IoU = 55.84
# ===> manmade - IoU = 45.84
# ===> vegetation - IoU = 38.72
# ===> mIoU of 6019 samples: 38.22
# {'mIoU': array([0.10152758, 0.46350998, 0.21222179, 0.41219927, 0.49650901,
#        0.23097634, 0.22441906, 0.23406273, 0.25591798, 0.32877811,
#        0.36742328, 0.81714909, 0.4239167 , 0.52463617, 0.5583566 ,
#        0.45837381, 0.38724564, 0.90546028])}

# 24 epoch
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 11.0
# ===> barrier - IoU = 47.22
# ===> bicycle - IoU = 22.15
# ===> bus - IoU = 41.38
# ===> car - IoU = 50.59
# ===> construction_vehicle - IoU = 23.19
# ===> motorcycle - IoU = 23.04
# ===> pedestrian - IoU = 23.81
# ===> traffic_cone - IoU = 26.21
# ===> trailer - IoU = 32.07
# ===> truck - IoU = 37.35
# ===> driveable_surface - IoU = 82.16
# ===> other_flat - IoU = 42.47
# ===> sidewalk - IoU = 53.27
# ===> terrain - IoU = 56.59
# ===> manmade - IoU = 46.7
# ===> vegetation - IoU = 39.71
# ===> mIoU of 6019 samples: 38.76
# {'mIoU': array([0.10995066, 0.47223532, 0.22150914, 0.41375673, 0.50592017,
#        0.2318567 , 0.23044849, 0.23808891, 0.2620557 , 0.32067361,
#        0.37348739, 0.82161153, 0.4247052 , 0.53268723, 0.56587486,
#        0.46699084, 0.39708123, 0.9040804 ])}
