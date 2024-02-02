_base_ = ['./bevstereo-occ-256x704.py']

model = dict(
    type='RenderOcc',
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
    ),
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

resume_from = "work_dirs/renderocc-7frame-256x704/epoch_17.pth"
runner = dict(type='EpochBasedRunner', max_epochs=24)

log_config = dict(
    interval=50,
)

# 12 epoch
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 10.1
# ===> barrier - IoU = 45.64
# ===> bicycle - IoU = 23.65
# ===> bus - IoU = 39.06
# ===> car - IoU = 49.05
# ===> construction_vehicle - IoU = 24.1
# ===> motorcycle - IoU = 21.25
# ===> pedestrian - IoU = 22.0
# ===> traffic_cone - IoU = 24.49
# ===> trailer - IoU = 31.75
# ===> truck - IoU = 35.9
# ===> driveable_surface - IoU = 81.03
# ===> other_flat - IoU = 39.8
# ===> sidewalk - IoU = 51.52
# ===> terrain - IoU = 55.3
# ===> manmade - IoU = 44.77
# ===> vegetation - IoU = 39.38
# ===> mIoU of 6019 samples: 37.57
# {'mIoU': array([0.10099931, 0.45636267, 0.23653947, 0.39057868, 0.49050112,
#        0.24099109, 0.21246329, 0.21996862, 0.24486159, 0.31752073,
#        0.35899616, 0.81029256, 0.39799262, 0.51517185, 0.55297571,
#        0.44765667, 0.39375391, 0.9051707 ])}

# 24 epoch
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 11.23
# ===> barrier - IoU = 46.09
# ===> bicycle - IoU = 23.56
# ===> bus - IoU = 41.36
# ===> car - IoU = 49.75
# ===> construction_vehicle - IoU = 25.75
# ===> motorcycle - IoU = 21.93
# ===> pedestrian - IoU = 23.23
# ===> traffic_cone - IoU = 25.25
# ===> trailer - IoU = 32.51
# ===> truck - IoU = 37.06
# ===> driveable_surface - IoU = 81.35
# ===> other_flat - IoU = 40.83
# ===> sidewalk - IoU = 52.19
# ===> terrain - IoU = 55.81
# ===> manmade - IoU = 45.66
# ===> vegetation - IoU = 40.19
# ===> mIoU of 6019 samples: 38.46
# {'mIoU': array([0.11233032, 0.46090279, 0.23563457, 0.41356269, 0.4974597 ,
#        0.25749657, 0.21930147, 0.23229435, 0.25249696, 0.32510802,
#        0.37057679, 0.81354295, 0.40832349, 0.52194215, 0.55814934,
#        0.45661405, 0.40187722, 0.90352228])}