# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet_occ import BEVStereo4DOCC
import torch.nn.functional as F
import torch
import cv2
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
from .. import builder
import time


# occ3d-nuscenes
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947, 
            497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031, 
            141625221, 2307405309])


@DETECTORS.register_module()
class RenderOcc(BEVStereo4DOCC):
    def __init__(self,
                 out_dim=32,
                 num_classes=18,
                 nerf_head=None,
                 test_threshold=8.5,
                 use_lss_depth_loss=True,
                 use_3d_loss=False,
                 balance_cls_weight=True,
                 final_softplus=False,
                 **kwargs):
        super(RenderOcc, self).__init__(use_predicter=False, **kwargs)
        self.out_dim = out_dim
        self.use_3d_loss = use_3d_loss
        self.test_threshold = test_threshold
        self.use_lss_depth_loss = use_lss_depth_loss
        self.balance_cls_weight = balance_cls_weight
        self.final_softplus = final_softplus

        if self.balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
            self.semantic_loss = nn.CrossEntropyLoss(
                    weight=self.class_weights, reduction="mean"
                )
        else:
            self.semantic_loss = nn.CrossEntropyLoss(reduction="mean")

        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        self.out_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))

        if self.final_softplus:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 2),
                nn.Softplus(),
            )
        else:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 2),
            )

        self.semantic_mlp = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2, num_classes-1),
        )

        if nerf_head is not None:
            self.nerf_head = builder.build_head(nerf_head)
        else:
            self.nerf_head = None
      

    # def loss_3d(self,voxel_semantics,mask_camera,density_prob, semantic):
    #     voxel_semantics=voxel_semantics.long()
    #
    #     voxel_semantics=voxel_semantics.reshape(-1)
    #     density_prob=density_prob.reshape(-1, 2)
    #     semantic = semantic.reshape(-1, self.num_classes-1)
    #     density_target = (voxel_semantics==17).long()
    #     semantic_mask = voxel_semantics!=17
    #
    #     # compute loss
    #     loss_geo=self.loss_occ(density_prob, density_target)
    #     loss_sem = self.semantic_loss(semantic[semantic_mask], voxel_semantics[semantic_mask].long())
    #
    #     loss_ = dict()
    #     loss_['loss_3d_geo'] = loss_geo
    #     loss_['loss_3d_sem'] = loss_sem
    #     return loss_

    def loss_3d(self, voxel_semantics, mask_camera, density_prob, semantic):
        voxel_semantics = voxel_semantics.long()

        voxel_semantics = voxel_semantics.reshape(-1)   # (B*Dx*Dy*Dz, )
        density_prob = density_prob.reshape(-1, 2)      # (B*Dx*Dy*Dz, 2)
        semantic = semantic.reshape(-1, self.num_classes - 1)
        density_target = (voxel_semantics == 17).long()
        semantic_mask = voxel_semantics != 17

        mask_camera = mask_camera.reshape(-1)
        num_total_samples = mask_camera.sum()
        # compute loss
        loss_geo = self.loss_occ(density_prob, density_target, mask_camera, avg_factor=num_total_samples)

        semantic_mask = torch.logical_and(semantic_mask, mask_camera)
        loss_sem = self.semantic_loss(semantic[semantic_mask], voxel_semantics[semantic_mask].long())

        loss_ = dict()
        loss_['loss_3d_geo'] = loss_geo
        loss_['loss_3d_sem'] = loss_sem
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        # extract volumn feature
        img_inputs = self.prepare_inputs(img, stereo=True)
        img_feats, _ = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        
        # predict SDF
        density_prob = self.density_mlp(voxel_feats)
        density = density_prob[..., 0]
        semantic = self.semantic_mlp(voxel_feats)

        # SDF --> Occupancy
        # no_empty_mask = density > self.test_threshold
        # semantic_res = semantic.argmax(-1)
        no_empty_mask = (density_prob.argmax(dim=-1) == 0)
        semantic_res = semantic.argmax(-1)
        
        B, H, W, Z, C = voxel_feats.shape
        occ = torch.ones((B,H,W,Z), dtype=semantic_res.dtype).to(semantic_res.device)
        occ = occ * (self.num_classes-1)
        occ[no_empty_mask] = semantic_res[no_empty_mask]

        occ = occ.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # extract volumn feature
        img_inputs = self.prepare_inputs(img_inputs, stereo=True)
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
                bda, curr2adjsensor = img_inputs
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        # (B, C=32, Dz, Dy, Dx) --> (B, C=32, Dz, Dy, Dx)
        # --> (B, Dx, Dy, Dz, C=32)
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc

        # predict SDF
        # (B, Dx, Dy, Dz, C=32) --> (B, Dx, Dy, Dz, C=64) --> (B, Dx, Dy, Dz, 2)
        density_prob = self.density_mlp(voxel_feats)
        density = density_prob[..., 0]      # (B, Dx, Dy, Dz)
        semantic = self.semantic_mlp(voxel_feats)   # (B, Dx, Dy, Dz, N_cls-1)

        # compute loss
        losses = dict()
        if self.use_3d_loss:      # 3D loss
            voxel_semantics = kwargs['voxel_semantics']
            mask_camera = kwargs['mask_camera']
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
            loss_occ = self.loss_3d(voxel_semantics, mask_camera, density_prob, semantic)
            losses.update(loss_occ)
        if self.nerf_head:          # 2D rendering loss
            loss_rendering = self.nerf_head(density, semantic, rays=kwargs['rays'], bda=bda)
            losses.update(loss_rendering)
        if self.use_lss_depth_loss: # lss-depth loss (BEVStereo's feature)
            loss_depth = self.img_view_transformer.get_depth_loss(kwargs['gt_depth'], depth)
            losses['loss_lss_depth'] = loss_depth
        return losses


# ----------------------------------------------------------------------------------------------------------------------
@DETECTORS.register_module()
class RenderOcc2D(BEVStereo4DOCC):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 num_classes=18,
                 nerf_head=None,
                 test_threshold=8.5,
                 use_lss_depth_loss=True,
                 use_3d_loss=False,
                 balance_cls_weight=True,
                 final_softplus=False,
                 **kwargs):
        super(RenderOcc2D, self).__init__(use_predicter=False, **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        self.num_classes = num_classes
        self.use_3d_loss = use_3d_loss
        self.test_threshold = test_threshold
        self.use_lss_depth_loss = use_lss_depth_loss
        self.balance_cls_weight = balance_cls_weight
        self.final_softplus = final_softplus

        if self.balance_cls_weight:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
            class_weights = class_weights / class_weights.sum()
            self.class_weights = class_weights
            self.semantic_loss = nn.CrossEntropyLoss(
                weight=self.class_weights, reduction="mean"
            )
        else:
            self.semantic_loss = nn.CrossEntropyLoss(reduction="mean")

        self.final_conv = ConvModule(
            self.in_dim,
            self.out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d'))

        if self.final_softplus:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, 2 * Dz),
                nn.Softplus(),
            )
        else:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, 2 * Dz),
            )

        self.semantic_mlp = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, (num_classes - 1) * Dz),
        )

        if nerf_head is not None:
            self.nerf_head = builder.build_head(nerf_head)
        else:
            self.nerf_head = None

    # def loss_3d(self, voxel_semantics, mask_camera, density_prob, semantic):
    #     voxel_semantics = voxel_semantics.long()
    #
    #     voxel_semantics = voxel_semantics.reshape(-1)
    #     density_prob = density_prob.reshape(-1, 2)
    #     semantic = semantic.reshape(-1, self.num_classes - 1)
    #     density_target = (voxel_semantics == 17).long()
    #     semantic_mask = voxel_semantics != 17
    #
    #     # compute loss
    #     loss_geo = self.loss_occ(density_prob, density_target)
    #     loss_sem = self.semantic_loss(semantic[semantic_mask], voxel_semantics[semantic_mask].long())
    #
    #     loss_ = dict()
    #     loss_['loss_3d_geo'] = loss_geo
    #     loss_['loss_3d_sem'] = loss_sem
    #     return loss_

    def loss_3d(self, voxel_semantics, mask_camera, density_prob, semantic):
        voxel_semantics = voxel_semantics.long()

        voxel_semantics = voxel_semantics.reshape(-1)   # (B*Dx*Dy*Dz, )
        density_prob = density_prob.reshape(-1, 2)      # (B*Dx*Dy*Dz, 2)
        semantic = semantic.reshape(-1, self.num_classes - 1)
        density_target = (voxel_semantics == 17).long()
        semantic_mask = voxel_semantics != 17

        mask_camera = mask_camera.reshape(-1)
        num_total_samples = mask_camera.sum()
        # compute loss
        loss_geo = self.loss_occ(density_prob, density_target, mask_camera, avg_factor=num_total_samples)

        semantic_mask = torch.logical_and(semantic_mask, mask_camera)
        loss_sem = self.semantic_loss(semantic[semantic_mask], voxel_semantics[semantic_mask].long())

        loss_ = dict()
        loss_['loss_3d_geo'] = loss_geo
        loss_['loss_3d_sem'] = loss_sem
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        # extract volumn feature
        img_inputs = self.prepare_inputs(img, stereo=True)
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        # (B, C=256, Dy, Dx) --> (B, C=256, Dy, Dx) --> (B, Dx, Dy, C)
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 3, 2, 1)

        B, Dx, Dy = voxel_feats.shape[:3]
        # predict SDF
        # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*2)
        density_prob = self.density_mlp(voxel_feats)
        density_prob = density_prob.view(B, Dx, Dy, self.Dz, 2)

        # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*(N_cls-1))
        semantic = self.semantic_mlp(voxel_feats)
        # (B, Dx, Dy, Dz*(N_cls-1)) --> (B, Dx, Dy, Dz, N_cls-1)
        semantic = semantic.view(B, Dx, Dy, self.Dz, self.num_classes - 1)

        # SDF --> Occupancy
        # density = density_prob[..., 0]  # (B, Dx, Dy, Dz)
        # no_empty_mask = density > self.test_threshold
        no_empty_mask = (density_prob.argmax(dim=-1) == 0)
        semantic_res = semantic.argmax(-1)

        B, H, W, Z, C = semantic.shape
        occ = torch.ones((B, H, W, Z), dtype=semantic_res.dtype).to(semantic_res.device)
        occ = occ * (self.num_classes - 1)
        occ[no_empty_mask] = semantic_res[no_empty_mask]

        occ = occ.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # extract volumn feature
        img_inputs = self.prepare_inputs(img_inputs, stereo=True)
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = img_inputs
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        # (B, C=256, Dy, Dx) --> (B, C=256, Dy, Dx) --> (B, Dx, Dy, C)
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 3, 2, 1)

        B, Dx, Dy = voxel_feats.shape[:3]
        # predict SDF
        # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*2)
        density_prob = self.density_mlp(voxel_feats)
        density_prob = density_prob.view(B, Dx, Dy, self.Dz, 2)
        density = density_prob[..., 0]   # (B, Dx, Dy, Dz)

        # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*(N_cls-1))
        semantic = self.semantic_mlp(voxel_feats)
        # (B, Dx, Dy, Dz*(N_cls-1)) --> (B, Dx, Dy, Dz, N_cls-1)
        semantic = semantic.view(B, Dx, Dy, self.Dz, self.num_classes-1)

        # compute loss
        losses = dict()
        if self.use_3d_loss:  # 3D loss
            voxel_semantics = kwargs['voxel_semantics']
            mask_camera = kwargs['mask_camera']
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
            loss_occ = self.loss_3d(voxel_semantics, mask_camera, density_prob, semantic)
            losses.update(loss_occ)
        if self.nerf_head:  # 2D rendering loss
            loss_rendering = self.nerf_head(density, semantic, rays=kwargs['rays'], bda=bda)
            losses.update(loss_rendering)
        if self.use_lss_depth_loss:  # lss-depth loss (BEVStereo's feature)
            loss_depth = self.img_view_transformer.get_depth_loss(kwargs['gt_depth'], depth)
            losses['loss_lss_depth'] = loss_depth
        return losses
