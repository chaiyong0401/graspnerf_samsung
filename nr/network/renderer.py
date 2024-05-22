import torch
import numpy as np
import torch.nn as nn

from network.aggregate_net import name2agg_net
from network.dist_decoder import name2dist_decoder
from network.init_net import name2init_net
from network.ops import ResUNetLight
from network.vis_encoder import name2vis_encoder
from network.render_ops import *
from utils.field_utils import TSDF_SAMPLE_POINTS

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class NeuralRayRenderer(nn.Module):
    base_cfg={
        'vis_encoder_type': 'default',
        'vis_encoder_cfg': {},

        'dist_decoder_type': 'mixture_logistics',
        'dist_decoder_cfg': {},

        'agg_net_type': 'default',
        'agg_net_cfg': {},

        'use_hierarchical_sampling': False,
        'fine_agg_net_cfg': {},
        'fine_dist_decoder_cfg': {},
        'fine_depth_sample_num': 64,
        'fine_depth_use_all': False,


        'ray_batch_num': 2048,
        'depth_sample_num': 64,
        'alpha_value_ground_state': -15,
        'use_dr_prediction': False,
        'use_nr_color_for_dr': False,
        'use_self_hit_prob': False,
        'use_ray_mask': True,
        'ray_mask_view_num': 2,
        'ray_mask_point_num': 8,

        'render_depth': False,
        'disable_view_dir': False,
        'render_rgb': False,

        'init_net_type': 'depth',
        'init_net_cfg': {},
        'depth_loss_coords_num': 8192,
    }
    def __init__(self,cfg):
        super().__init__()
        self.cfg = {**self.base_cfg, **cfg}
        self.vis_encoder = name2vis_encoder[self.cfg['vis_encoder_type']](self.cfg['vis_encoder_cfg'])
        self.dist_decoder = name2dist_decoder[self.cfg['dist_decoder_type']](self.cfg['dist_decoder_cfg'])
        self.image_encoder = ResUNetLight(3, [1,2,6,4], 32, inplanes=16)
        self.init_net=name2init_net[self.cfg['init_net_type']](self.cfg['init_net_cfg'])
        self.agg_net = name2agg_net[self.cfg['agg_net_type']](self.cfg['agg_net_cfg'])
        if self.cfg['use_hierarchical_sampling']:
            self.fine_dist_decoder = name2dist_decoder[self.cfg['dist_decoder_type']](self.cfg['fine_dist_decoder_cfg'])
            self.fine_agg_net = name2agg_net[self.cfg['agg_net_type']](self.cfg['fine_agg_net_cfg'])

        self.use_sdf = self.cfg['agg_net_type'] in ['neus']

    def predict_proj_ray_prob(self, prj_dict, ref_imgs_info, que_dists, is_fine):
        rfn, qn, rn, dn, _ = prj_dict['mask'].shape
        # decode ray prob
        if is_fine:
            prj_mean, prj_var, prj_vis, prj_aw = self.fine_dist_decoder(prj_dict['ray_feats'])
        else:
            prj_mean, prj_var, prj_vis, prj_aw = self.dist_decoder(prj_dict['ray_feats'])

        alpha_values, visibility, hit_prob = self.dist_decoder.compute_prob(
            prj_dict['depth'].squeeze(-1),que_dists.unsqueeze(0),prj_mean,prj_var,
            prj_vis, prj_aw, True, ref_imgs_info['depth_range'])
        # post process
        prj_dict['alpha'] = alpha_values.reshape(rfn,qn,rn,dn,1) * prj_dict['mask'] + \
                            (1 - prj_dict['mask']) * self.cfg['alpha_value_ground_state']
        prj_dict['vis'] = visibility.reshape(rfn,qn,rn,dn,1) * prj_dict['mask']
        prj_dict['hit_prob'] = hit_prob.reshape(rfn,qn,rn,dn,1) * prj_dict['mask']
        return prj_dict

    def get_img_feats(self,ref_imgs_info, prj_dict):
        rfn, _, h, w = ref_imgs_info['imgs'].shape
        rfn, qn, rn, dn, _ = prj_dict['pts'].shape

        img_feats = ref_imgs_info['img_feats']
        prj_img_feats = interpolate_feature_map(img_feats, prj_dict['pts'].reshape(rfn, qn * rn * dn, 2),
                                                prj_dict['mask'].reshape(rfn, qn * rn * dn), h, w,)
        prj_dict['img_feats'] = prj_img_feats.reshape(rfn, qn, rn, dn, -1)
        return prj_dict

    # 신경망을 이용하여 3D 장면 또는 객체의 렌더링을 수행하고, 선택적으로 SDF 관련 값을 계산하는 과정을 정의
    def network_rendering(self, prj_dict, que_dir, que_pts, que_depth, is_fine, is_train, is_sdf=False, sdf_only=False): 
        # prj_dict(투영 정보를 담고 있는 딕셔너리), que_dir(쿼리 방향(레이 방향, 시야)), que_pts(쿼리 포인트 위치, 3D 공간 포인트), 
        # que_depth(쿼리 포인트들의 깊이 정보 텐서, 거리 반환을 계산하는데 사용), is_fine(세밀한 렌더링)
        net = self.fine_agg_net if is_fine else self.agg_net
        que_dists = depth2dists(que_depth) if que_depth is not None else None   # 깊이 정보를 바탕으로 거리를 계산
        rendering_outputs = net(prj_dict, que_dir, que_pts, que_dists, is_train)    # 네트워크에 입력값들을 전달하여 렌더링 수행
        outputs = {}
        if is_sdf:
            alpha_values, outputs['sdf_values'], colors, outputs['sdf_gradient_error'], outputs['s'] = rendering_outputs
            # outputs[sdf_values] size = [1,1600,40]
            print("sdf size")
            print(outputs['sdf_values'].shape)
            if sdf_only:
                return outputs
        else:
            density, colors = rendering_outputs
            alpha_values = 1.0 - torch.exp(-torch.relu(density))

        outputs['alpha_values'] = alpha_values  #포인트의 불투명도
        outputs['colors_nr'] = colors   
        outputs['hit_prob_nr'] = hit_prob = alpha_values2hit_prob(alpha_values)
        outputs['pixel_colors_nr'] = torch.sum(hit_prob.unsqueeze(-1)*colors,2)

        return outputs

    def render_by_depth(self, que_depth, que_imgs_info, ref_imgs_info, is_train, is_fine):
        ref_imgs_info = ref_imgs_info.copy()
        que_imgs_info = que_imgs_info.copy()
        que_dists = depth2inv_dists(que_depth,que_imgs_info['depth_range'])
        # generate points with query depth
        que_pts, que_dir = depth2points(que_imgs_info, que_depth)
        if self.cfg['disable_view_dir']:
            que_dir = None
        prj_dict = project_points_dict(ref_imgs_info, que_pts)
        prj_dict = self.predict_proj_ray_prob(prj_dict, ref_imgs_info, que_dists, is_fine)
        prj_dict = self.get_img_feats(ref_imgs_info, prj_dict)

        outputs = self.network_rendering(prj_dict, que_dir, que_pts, que_depth, is_fine, is_train, is_sdf=self.use_sdf) 


        if 'imgs' in que_imgs_info:
            outputs['pixel_colors_gt'] = interpolate_feats(
                que_imgs_info['imgs'], que_imgs_info['coords'], align_corners=True)

        if self.cfg['use_ray_mask']:
            outputs['ray_mask'] = torch.sum(prj_dict['mask'].int(),0)>self.cfg['ray_mask_view_num'] # qn,rn,dn,1
            outputs['ray_mask'] = torch.sum(outputs['ray_mask'],2)>self.cfg['ray_mask_point_num'] # qn,rn
            outputs['ray_mask'] = outputs['ray_mask'][...,0]

        if self.cfg['render_depth']:
            # qn,rn,dn
            outputs['render_depth'] = torch.sum(outputs['hit_prob_nr'] * que_depth, -1) # qn,rn
            #outputs['render_depth_dr'] = torch.sum(hit_prob_dr * que_depth, -1) # qn,rn
        return outputs

    def fine_render_impl(self, coarse_render_info, que_imgs_info, ref_imgs_info, is_train):
        fine_depth = sample_fine_depth(coarse_render_info['depth'], coarse_render_info['hit_prob'].detach(),
                                       que_imgs_info['depth_range'], self.cfg['fine_depth_sample_num'], is_train)

        # qn, rn, fdn+dn
        if self.cfg['fine_depth_use_all']:
            que_depth = torch.sort(torch.cat([coarse_render_info['depth'], fine_depth], -1), -1)[0]
        else:
            que_depth = torch.sort(fine_depth, -1)[0]
        outputs = self.render_by_depth(que_depth, que_imgs_info, ref_imgs_info, is_train, True)
        return outputs

    def render_impl(self, que_imgs_info, ref_imgs_info, is_train):
        # [qn,rn,dn]
        # sample points along test ray at different depth
        que_depth, _ = sample_depth(que_imgs_info['depth_range'], que_imgs_info['coords'], self.cfg['depth_sample_num'], False)
        outputs = self.render_by_depth(que_depth, que_imgs_info, ref_imgs_info, is_train, False)
        if self.cfg['use_hierarchical_sampling']:
            coarse_render_info= {'depth': que_depth, 'hit_prob': outputs['hit_prob_nr']}
            fine_outputs = self.fine_render_impl(coarse_render_info, que_imgs_info, ref_imgs_info, is_train)
            for k, v in fine_outputs.items():
                outputs[k + "_fine"] = v
        return outputs

    def sample_volume(self, ref_imgs_info):
        ref_imgs_info = ref_imgs_info.copy()    # 참조 이미지들과 관련 정보를 포함하는 딕셔너리
        res = self.cfg['volume_resolution']
        que_pts = ( torch.from_numpy(TSDF_SAMPLE_POINTS).to(ref_imgs_info['imgs'].device) +     # 볼륨 내의 쿼리 포인트들 생성. 해상도에 따라 3D 공간에 균일하게 분포
                    torch.tensor(ref_imgs_info['bbox3d'][0], device=ref_imgs_info['imgs'].device)
                ).reshape(1, res * res, res, 3)
        que_pts = torch.flip(que_pts, (2,)) # z축에 따라 뒤집혀 볼륨의 상단부터 하단까지 샘플링

        prj_dict = project_points_dict(ref_imgs_info, que_pts)
        prj_dict = self.get_img_feats(ref_imgs_info, prj_dict)
        valid_ratio = torch.sum(prj_dict['mask'],dim=(1,2,3,4)) / np.prod(list(prj_dict['mask'].shape)[1:])
        if torch.mean(valid_ratio) < 0.5:
            print("!! too low ratio", valid_ratio)
        
        prj_dict = self.predict_proj_ray_prob(prj_dict, ref_imgs_info, torch.empty(0), False)
        que_dir = torch.tensor([0,0,1], device=que_pts.device).reshape(1,1,1,3).repeat(1, res * res, res, 1) if not self.cfg['disable_view_dir'] else None

        feat_list = []
        mode = self.cfg['volume_type']
        if 'image' in mode:
            image_feat = torch.cat([prj_dict['rgb'], prj_dict['img_feats']], dim=-1)
            mean = torch.mean(image_feat, dim=-1)
            var = torch.var(image_feat, dim=-1)
            feat_list.append(torch.cat([image_feat, mean, var], dim=-1).reshape(1, res, res, res, -1).permute(1,-1))

        if 'alpha' in mode:
            outputs = self.network_rendering(prj_dict, que_dir, que_pts, None, False, False)
            feat_list.append(outputs['alpha_values'].reshape(1, 1, res, res, res))
        
        if 'sdf' in mode:
            ##### 실제 적용 mode ###################
            ## output이 원래의 sdf랑 모양이 많이 달라짐 ##
            outputs = self.network_rendering(prj_dict, que_dir, que_pts, None, False, False, is_sdf=self.use_sdf, sdf_only=True)
            feat_list.append(outputs['sdf_values'].reshape(1, 1, res, res, res))

        feat = torch.cat(feat_list, dim=1)
        feat = torch.flip(feat, (-1,)) # we sample from top to down, so we need to flip here
        center_region_sdf = feat[0, 0, res//2-5:res//2+5, res//2-5:res//2+5, res//2-5:res//2+5]
        print(center_region_sdf)
        sdf_values = feat.cpu().numpy()
        grid = sdf_values[0,0]

        x,y,z = np.indices(grid.shape)
        z_min_interest = 0
        z_max_interest = 40
        max_size = 40

        # 기존의 마스크 조건에 z의 관심 범위를 추가합니다.
        # mask = (z >= z_min_interest) & (z <= z_max_interest) & (grid < 0.01) & (grid >-0.8)
        mask = (grid < 0.01) & (grid >-0.8)
        masked_x, masked_y, masked_z = x[mask], y[mask], z[mask]
        masked_xyz = np.vstack((masked_x, masked_y, masked_z)).T

        scaled_xyz = np.floor((masked_xyz - np.min(masked_xyz, axis=0)) / (np.max(masked_xyz, axis=0) - np.min(masked_xyz, axis=0)) * (max_size - 1))
        voxel_grid = np.zeros((max_size, max_size, max_size), dtype=bool)

        for point in scaled_xyz:
            x, y, z = point.astype(int)
            voxel_grid[x, y, z] = True

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = np.indices((max_size + 1, max_size + 1, max_size + 1))
        ax.voxels(x, y, z, voxel_grid, edgecolor='k')

        ax.set_box_aspect([1, 1, 1])
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x[mask], y[mask], z[mask], alpha=0.6)

        # 축 레이블 설정
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 시각화 표시
        # plt.show()


        return feat

    def render(self, que_imgs_info, ref_imgs_info, is_train):
        render_info_all = {} 
        ray_batch_num = self.cfg["ray_batch_num"]
        coords = que_imgs_info['coords']
        ray_num = coords.shape[1]
        
        for ray_id in range(0,ray_num,ray_batch_num):
            que_imgs_info['coords']=coords[:,ray_id:ray_id+ray_batch_num]
            render_info = self.render_impl(que_imgs_info,ref_imgs_info,is_train)
            output_keys = [k for k in render_info.keys()]
            for k in output_keys:
                v = render_info[k]
                if k not in render_info_all:
                    render_info_all[k]=[]
                render_info_all[k].append(v)

        for k, v in render_info_all.items():
            render_info_all[k]=torch.cat(v,1)

        return render_info_all

    def gen_depth_loss_coords(self,h,w,device):
        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).reshape(-1, 2).to(device)
        num = self.cfg['depth_loss_coords_num']
        idxs = torch.randperm(coords.shape[0])
        idxs = idxs[:num]
        coords = coords[idxs]
        return coords

    def predict_mean_for_depth_loss(self, ref_imgs_info):
        ray_feats = ref_imgs_info['ray_feats'] # rfn,f,h',w'
        ref_imgs = ref_imgs_info['imgs'] # rfn,3,h,w
        rfn, _, h, w = ref_imgs.shape
        coords = self.gen_depth_loss_coords(h,w,ref_imgs.device) # pn,2
        coords = coords.unsqueeze(0).repeat(rfn,1,1) # rfn,pn,2

        batch_num = self.cfg['depth_loss_coords_num']
        pn = coords.shape[1]
        coords_dist_mean, coords_dist_mean_2, coords_dist_mean_fine, coords_dist_mean_fine_2 = [], [], [], []
        for ci in range(0, pn, batch_num):
            coords_ = coords[:,ci:ci+batch_num]
            mask_ = torch.ones(coords_.shape[:2], dtype=torch.float32, device=ref_imgs.device)
            coords_ray_feats_ = interpolate_feature_map(ray_feats, coords_, mask_, h, w) # rfn,pn,f
            coords_dist_mean_ = self.dist_decoder.predict_mean(coords_ray_feats_)  # rfn,pn
            coords_dist_mean_2.append(coords_dist_mean_[..., 1])
            coords_dist_mean_ = coords_dist_mean_[..., 0]

            coords_dist_mean.append(coords_dist_mean_)
            if self.cfg['use_hierarchical_sampling']:
                coords_dist_mean_fine_ = self.fine_dist_decoder.predict_mean(coords_ray_feats_)
                coords_dist_mean_fine_2.append(coords_dist_mean_fine_[..., 1])
                coords_dist_mean_fine_ = coords_dist_mean_fine_[..., 0]  # use 0 for depth supervision
                coords_dist_mean_fine.append(coords_dist_mean_fine_)

        coords_dist_mean = torch.cat(coords_dist_mean, 1)
        outputs = {'depth_mean': coords_dist_mean, 'depth_coords': coords}
        if len(coords_dist_mean_2)>0:
            coords_dist_mean_2 = torch.cat(coords_dist_mean_2, 1)
            outputs['depth_mean_2'] = coords_dist_mean_2
        if self.cfg['use_hierarchical_sampling']:
            coords_dist_mean_fine = torch.cat(coords_dist_mean_fine, 1)
            outputs['depth_mean_fine'] = coords_dist_mean_fine
            if len(coords_dist_mean_fine_2)>0:
                coords_dist_mean_fine_2 = torch.cat(coords_dist_mean_fine_2, 1)
                outputs['depth_mean_fine_2'] = coords_dist_mean_fine_2
        return outputs

    def forward(self,data):
        ref_imgs_info = data['ref_imgs_info'].copy()
        que_imgs_info = data['que_imgs_info'].copy()
        is_train = 'eval' not in data
        src_imgs_info = data['src_imgs_info'].copy() if 'src_imgs_info' in data else None

        # extract image feature
        ref_imgs_info['img_feats'] = self.image_encoder(ref_imgs_info['imgs'])
        # calc visibility feature map of each view from mvs
        ref_imgs_info['ray_feats'] = self.init_net(ref_imgs_info, src_imgs_info, is_train)
        # refine visibity feature along with image feature
        ref_imgs_info['ray_feats'] = self.vis_encoder(ref_imgs_info['ray_feats'], ref_imgs_info['img_feats'])
        
        render_outputs = {}
        if self.cfg['render_rgb']:
            render_outputs = self.render(que_imgs_info, ref_imgs_info, is_train)

        if self.cfg['sample_volume']:
            render_outputs['volume'] = self.sample_volume(ref_imgs_info)

        if (self.cfg['use_depth_loss'] and 'true_depth' in ref_imgs_info) or (not is_train):
            render_outputs.update(self.predict_mean_for_depth_loss(ref_imgs_info))

        return render_outputs

class GraspNeRF(nn.Module):
    default_cfg_vgn={                   # VGN 기본 설정을 담은 딕셔너리, 초기 학습 단계 수와 초기 학습 이후 NeRF 모델 동결 여부 결정
        'nr_initial_training_steps': 0,
        'freeze_nr_after_init': False
    }
    def __init__(self, cfg):
        super().__init__()
        self.cfg={**self.default_cfg_vgn,**cfg}
        from gd.networks import get_network
        self.nr_net = NeuralRayRenderer(self.cfg)   # NeRF 네트워크 초기화. 
        self.vgn_net = get_network("conv")          # VGN 네트워크 구조를 가져온다. convolution network: 

    def select(self, out, index):      # 출력 tensor에서 특정 인덱스에 해당하는 요소를 선택하여 봔환
        qual_out, rot_out, width_out = out
        batch_index = torch.arange(qual_out.shape[0])
        label = qual_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]].squeeze()
        rot = rot_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]]
        width = width_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]].squeeze()
        return (label, rot, width)

    def forward(self, data):
        if data['step'] < self.cfg['nr_initial_training_steps']:    # data 딕셔너래 내부의 step 키의 값에 따라 초기 학습 단계인지 여부를 확인하고 초기 학습 단계에만 NeRF 모델 사용하거나, 설정에 따라 항상 NeRF 사용
            render_outputs = super().forward(data)
            with torch.no_grad():
                vgn_pred = self.vgn_net(render_outputs['volume'])
        elif self.cfg['freeze_nr_after_init']:
            with torch.no_grad():
                render_outputs = super().forward(data)
            vgn_pred = self.vgn_net(render_outputs['volume'])
        else:
            ### 실제로 실행 되는 코드 ###
            render_outputs = self.nr_net(data)    #  nr_net을 통해 tsdf 볼륨 생성 
            vgn_pred = self.vgn_net(render_outputs['volume'])   # 생성된 volume을 vgn(grasp detection network)에 넣어 qual,rot,width를 얻는다. 

        
        if 'full_vol' not in data: 
            print("not_full_vol")
            render_outputs['vgn_pred'] = self.select(vgn_pred, data['grasp_info'][0]) # select 함수를 이용하여 grasp_info에서 제공된 인덱스에 따라 VGN의 출력에서 필요한 정보를 선택하거나 full_vol 반환
        else:
            render_outputs['vgn_pred'] = vgn_pred
            q,r,w = vgn_pred
            print(q.shape)  # [1,1,40,40,40]
            print(r.shape)  # [1,4,40,40,40]
            print(w.shape)  # [1,1,40,40,40]

        return render_outputs

name2network={
    'grasp_nerf': GraspNeRF,
}
