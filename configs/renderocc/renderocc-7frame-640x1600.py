_base_ = ['./bevstereo-occ-640x1600.py']

model = dict(
    type='RenderOcc',
    final_softplus=True,
    nerf_head=dict(
        type='NerfHead',
        point_cloud_range= [-40,-40,-1, 40,40,5.4],
        voxel_size=0.4,
        scene_center=[0, 0, 2.2],
        radius=39,
        use_depth_sup=True,
    )
)

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)

depth_gt_path = 'data/nuscenes/depth_gt'
semantic_gt_path = 'data/nuscenes/seg_gt_lidarseg'

data = dict(
    samples_per_gpu=1,  # with 8 GPU, Batch Size=16
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


