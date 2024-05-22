import sys, os
import time

sys.path.append("./src/nr")
from pathlib import Path
import numpy as np

import torch
from skimage.io import imsave, imread
from network.renderer import name2network
from utils.base_utils import load_cfg, to_cuda
from utils.imgs_info import build_render_imgs_info, imgs_info_to_torch, grasp_info_to_torch
from network.renderer import name2network
from utils.base_utils import color_map_forward
from network.loss import VGNLoss
from tqdm import tqdm
from scipy import ndimage
import cv2
from gd.utils.transform import Transform, Rotation
from gd.grasp import *


def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=1.33,
    max_width=9.33,
    tsdf_thres_high = 0.5,
    tsdf_thres_low = 1e-3,
    n_grasp=0
):
    tsdf_vol = tsdf_vol.squeeze()  
    qual_vol = qual_vol.squeeze()  
    rot_vol = rot_vol.squeeze()  
    width_vol = width_vol.squeeze()
    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > tsdf_thres_high
    inside_voxels = np.logical_and(tsdf_thres_low < tsdf_vol, tsdf_vol < tsdf_thres_high)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0
    
    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4):    # 주어진 볼륨 데이터를 이용하여 grasp 선택하는 과정 구현.
    qual_vol[qual_vol < threshold] = 0.0    # qual_vol에서 임계값 이하의 값은 모두 '0'으로 변경

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)    # qual_vol 내부에서 지역 최대값을 찾는다. max_filter_size는 이웃하는 영역의 크기 정의 
    
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0) # 원본 qual_vol과 최대값으로 필터링 된 max_vol을 비교하여 최대값이 아닌 영역의 값을 0으로 설정한다. 
    mask = np.where(qual_vol, 1.0, 0.0) # qual_vol의 요소가 0이 아닌 경우에는 1을, 0인 경우에는 0을 가지는 배열 'mask' 
    print(mask.shape)
    # construct grasps
    grasps, scores, indexs = [], [], []
    for index in np.argwhere(mask): # qual_vol에서 0이 아닌 위치를 순회하며, 해당 위치에서 grasp, score 계산
        indexs.append(index)
        grasp, score = select_index(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)
    return grasps, scores, indexs


def select_index(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    rot = rot_vol[:, i, j, k]
    ori = Rotation.from_quat(rot)
    pos = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score


class GraspNeRFPlanner(object):
    def set_params(self, args):
        self.args = args
        self.voxel_size = 0.3 / 40
        self.bbox3d =  [[-0.15, -0.15, -0.0503],[0.15, 0.15, 0.2497]]   # boundary box의 최소 및 최대 좌표
        self.tsdf_thres_high = 0 
        self.tsdf_thres_low = -0.85

        self.renderer_root_dir = self.args.renderer_root_dir
        tp, split, scene_type, scene_split, scene_id, background_size = args.database_name.split('/')
        background, size = background_size.split('_')
        self.split = split
        self.tp = tp
        self.downSample = float(size) 
        tp2wh = {
            'vgn_syn': (640, 360)
        }
        src_wh = tp2wh[tp]
        self.img_wh = (np.array(src_wh) * self.downSample).astype(int)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.K = np.array([[892.62, 0.0, 639.5],
                           [0.0, 892.62, 359.5],
                           [0.0, 0.0, 1.0]]) 
        self.K[:2] = self.K[:2] * self.downSample
        if self.tp == 'vgn_syn':
            self.K[:2] /= 2
        self.depth_thres = {
            'vgn_syn': 0.8,
        }
        
        if args.object_set == "graspnet":
            dir_name = "pile_graspnet_test"
        else:
            if self.args.scene == "pile":
                dir_name = "pile_pile_test_200"
            elif self.args.scene == "packed":
                dir_name = "packed_packed_test_200"
            elif self.args.scene == "single":
                dir_name = "single_single_test_200"

        scene_root_dir = os.path.join(self.renderer_root_dir, "data/mesh_pose_list", dir_name)  # 현재 설정에 따라 적절한 시나리오 디렉토리 결정
        self.mesh_pose_list = [i for i in sorted(os.listdir(scene_root_dir))]
        self.depth_root_dir = ""
        self.depth_list = []

    def __init__(self, args=None, cfg_fn=None, debug_dir=None) -> None:
        default_render_cfg = {
        'min_wn': 3, # working view number
        'ref_pad_interval': 16, # input image size should be multiple of 16
        'use_src_imgs': False, # use source images to construct cost volume or not
        'cost_volume_nn_num': 3, # number of source views used in cost volume
        'use_depth': True, # use colmap depth in rendering or not,
        }
        # load render cfg
        if cfg_fn is None:
            self.set_params(args)
            cfg = load_cfg(args.cfg_fn)
        else:
            cfg = load_cfg(cfg_fn)

        print(f"[I] GraspNeRFPlanner: using ckpt: {cfg['name']}")
        render_cfg = cfg['train_dataset_cfg'] if 'train_dataset_cfg' in cfg else {}
        render_cfg = {**default_render_cfg, **render_cfg}
        cfg['render_rgb'] = False # only for training. Disable in grasping.
        # load model
        self.net = name2network[cfg['network']](cfg)
        ckpt_filename = 'model_best'
        # ckpt_filename = 'model_best_0119'
        ckpt = torch.load(Path('src/nr/ckpt') / cfg["group_name"] / cfg["name"] / f'{ckpt_filename}.pth')
        self.net.load_state_dict(ckpt['network_state_dict'])
        self.net.cuda()
        self.net.eval()
        self.step = ckpt["step"]
        self.output_dir = debug_dir
        if debug_dir is not None:
            if not Path(debug_dir).exists():
                Path(debug_dir).mkdir(parents=True)
        self.loss = VGNLoss({})
        self.num_input_views = render_cfg['num_input_views']
        print(f"[I] GraspNeRFPlanner: load model at step {self.step} of best metric {ckpt['best_para']}")

    def get_image(self, img_id, round_idx):
        img_filename = os.path.join(self.args.log_root_dir, "rendered_results/" + str(self.args.logdir).split("/")[-1], "rgb/%04d.png"%img_id)
        img = imread(img_filename)[:,:,:3]
        img = cv2.resize(img, self.img_wh)
        return np.asarray(img, dtype=np.float32)
    
    def get_pose(self, img_id): # img_id에 해당하는 카메라의 포즈(카메라의 world 좌표계에 대한 위치와 방향) 
        poses_ori = np.load(Path(self.renderer_root_dir) / 'camera_pose.npy')
        poses = [np.linalg.inv(p @ self.blender2opencv)[:3,:] for p in poses_ori]
        return poses[img_id].astype(np.float32).copy()
    
    def get_K(self, img_id):  
        return self.K.astype(np.float32).copy()

    def get_depth_range(self,img_id, round_idx, fixed=False):
        if fixed:
            return np.array([0.2,0.8])
        depth = self.get_depth(img_id, round_idx)
        nf = [max(0, np.min(depth)), min(self.depth_thres[self.tp], np.max(depth))]
        return np.array(nf)
    
    def __call__(self, test_view_id, round_idx, n_grasp, gt_tsdf):
        # load data for test
        images = [self.get_image(i, round_idx) for i in test_view_id]
        images = color_map_forward(np.stack(images, 0)).transpose([0, 3, 1, 2])
        extrinsics = np.stack([self.get_pose(i) for i in test_view_id], 0)      
        intrinsics = np.stack([self.get_K(i) for i in test_view_id], 0)
        depth_range = np.asarray([self.get_depth_range(i, round_idx, fixed = True) for i in test_view_id], dtype=np.float32)
        ############################ core 함수 변경을 통해 swindrnet으로 만들어진 tsdf_vol에 qual, rot, width 필요, core 함수가 기존 vgn의 predict에 해당############
        tsdf_vol, qual_vol_ori, rot_vol_ori, width_vol_ori, toc = self.core(images, extrinsics, intrinsics, depth_range, self.bbox3d)
        qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol_ori, rot_vol_ori, width_vol_ori, tsdf_thres_high=self.tsdf_thres_high, tsdf_thres_low=self.tsdf_thres_low, n_grasp=n_grasp)
        grasps, scores, indexs = select(qual_vol.copy(), rot_vol, width_vol)    # qual_vol.copy()를 통해 원본 데이터를 유지하고 변형시키지 않기 위해 qual_vol 복사본 생성해서 이용
        grasps, scores, indexs = np.asarray(grasps), np.asarray(scores), np.asarray(indexs)

        print(f"score: {scores}")
        print(f"grasps: {grasps}")
        if len(grasps) > 0:
            # grasp와 이에 대응하는 score, indxe 세트를 random하게 섞어준다.
            np.random.seed(self.args.seed + round_idx + n_grasp)
            p = np.random.permutation(len(grasps))  # grasp list의 길이와 같은 정수 시퀸스를 무작위로 섞어서 새로운 순열 'p' 생성
            grasps = [from_voxel_coordinates(g, self.voxel_size) for g in grasps[p]]    # random하게 섞인 순서 'p'에 따라 grasp list를 재정렬하고, 각 잡기 포즈'g'를 voxel 좌표게에서 실제 좌표계로 변환. 각 잡기 포즈를 실제 크기로 스케일링하는 데 필요한 self.voxel_size 사용
            scores = scores[p]  # score와 index 리스트도 순열 'p'에 따라 재정렬 -> grasp 리스트와 동일한 순서 유지 
            indexs = indexs[p]
            
        return grasps, scores, toc
    

    ### 이미지, 카메라 외부 매개변수, 내부 매개변수를 입력으로 받아 3D 볼륨 data와 그리핑 관련 정보를 생성 하는 함수 ##############
    def core(self, 
                images: np.ndarray, 
                extrinsics: np.ndarray, 
                intrinsics: np.ndarray, 
                depth_range=[0.2, 0.8], 
                bbox3d=[[-0.15, -0.15, -0.05],[0.15, 0.15, 0.25]], gt_info=None, que_id=0):
        """
        @args
            images: np array of shape (3, 3, h, w), image in RGB format
            extrinsics: np array of shape (3, 4, 4), the transformation matrix from world to camera, 3D 장면의 한 점이 카메라의 시야에서 어떻게 나타나는지 결정 
            intrinsics: np array of shape (3, 3, 3). 3D 카메라 좌표계에서 2D 이미지 픽셀 좌표계로 포인트를 투영하는데 사용. 
        @rets
            volume, label, rot, width: np array of shape (1, 1, res, res, res)
        """
        _, _, h, w = images.shape
        assert h % 32 == 0 and w % 32 == 0      #이미지(rgb)의 높이와 너비가 32의 배수인지 확인하여, 이미지 처리 시 요구되는 조건 만족
        extrinsics = extrinsics[:, :3, :]
        que_imgs_info = build_render_imgs_info(extrinsics[que_id], intrinsics[que_id], (h, w), depth_range[que_id])
        src_imgs_info = {'imgs': images, 'poses': extrinsics.astype(np.float32), 'Ks': intrinsics.astype(np.float32), 'depth_range': depth_range.astype(np.float32), 
                                'bbox3d': np.array(bbox3d)}

        ref_imgs_info = src_imgs_info.copy()
        num_views = images.shape[0]
        ref_imgs_info['nn_ids'] = np.arange(num_views).repeat(num_views, 0)
        data = {'step': self.step , 'eval': True}
        if not gt_info:
            data['full_vol'] = True
        else:
            data['grasp_info'] = to_cuda(grasp_info_to_torch(gt_info))
        data['que_imgs_info'] = to_cuda(imgs_info_to_torch(que_imgs_info))
        data['src_imgs_info'] = to_cuda(imgs_info_to_torch(src_imgs_info))
        data['ref_imgs_info'] = to_cuda(imgs_info_to_torch(ref_imgs_info))

        with torch.no_grad():
            t0 = time.time()
            render_info = self.net(data)
            t = time.time() - t0
        
        if gt_info:
            return self.loss(render_info, data, self.step, False)

        label, rot, width = render_info['vgn_pred']
        
        return render_info['volume'].cpu().numpy(), label.cpu().numpy(), rot.cpu().numpy(), width.cpu().numpy(), t
    
    ########## vgn에서 가져온 tsdf_vol을 이용해서 qual_vol, rot_vol, width_vol을 예측하는 함수
    def predict(tsdf_vol, net, device):
        assert tsdf_vol.shape == (1, 40, 40, 40)

        # move input to the GPU
        tsdf_vol = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)

        # forward pass
        with torch.no_grad():
            qual_vol, rot_vol, width_vol = net(tsdf_vol)

        # move output back to the CPU
        qual_vol = qual_vol.cpu().squeeze().numpy()
        rot_vol = rot_vol.cpu().squeeze().numpy()
        width_vol = width_vol.cpu().squeeze().numpy()
        return qual_vol, rot_vol, width_vol