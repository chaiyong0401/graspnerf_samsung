from math import cos, sin
import time

import numpy as np
import open3d as o3d

from gd.utils.transform import Transform

import matplotlib.pyplot as plt

from PIL import Image
import Imath
import OpenEXR
class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy, channel=1):
        self.width = width
        self.height = height
        self.channel = channel
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "channel": self.channel,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            channel=data["channel"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
        )
        return intrinsic


class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

    def integrate(self, depth_img, intrinsic, extrinsic):
        """
        Args:
            depth_img: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
        """
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.empty_like(depth_img)),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )


        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_grid(self):
        cloud = self._volume.extract_voxel_point_cloud()
        points = np.asarray(cloud.points)
        distances = np.asarray(cloud.colors)[:, [0]]
        grid = np.zeros((1, 40, 40, 40), dtype=np.float32)
        for idx, point in enumerate(points):
            i, j, k = np.floor(point / self.voxel_size).astype(int)
            grid[0, i, j, k] = distances[idx]
        return grid

    def get_cloud(self):
        return self._volume.extract_point_cloud()


def create_tsdf(size, resolution, depth_imgs, intrinsic, extrinsics):
    tsdf = TSDFVolume(size, resolution)
    for i in range(depth_imgs.shape[0]):
        extrinsic = Transform.from_list(extrinsics[i])
        tsdf.integrate(depth_imgs[i], intrinsic, extrinsic)
    return tsdf


def camera_on_sphere(origin, radius, theta, phi):
    eye = np.r_[
        radius * sin(theta) * cos(phi),
        radius * sin(theta) * sin(phi),
        radius * cos(theta),
    ]
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
    return Transform.look_at(eye, target, up) * origin.inverse()


def save_rgb_image(rgb_image,file_path):
    if rgb_image is not None and rgb_image.size != 0 :
        image = Image.fromarray(rgb_image.astype('uint8'),'RGB')
        image.save(file_path)
        print(f"image saved to {file_path}")
    else: 
        print("fail to save rgb image")
    
def save_depth_image(depth_img,file_path):
    depth_img = depth_img.astype(np.float32)    # 깊이 데이터 타입을 float32로 변환
    width, height = depth_img.shape[1], depth_img.shape[0]

    # OpenEXR 파일로 저장
    header = OpenEXR.Header(width, height) # openEXR 파일의 헤더를 생성. 이미지의 너비와 높이는 depth_image.shape[1], depth_image.shape[0] 으로 지정
    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) # 이미지 채널의 픽셀 타입을 'float'로 설정. 
    header['channels'] = {'R': float_chan, 'G': float_chan, 'B': float_chan}  # 헤더에 RGB 채널 정보 추가 
    header['compression'] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    header['dataWindow'] = Imath.Box2i(Imath.V2i(0, 0), Imath.V2i(width - 1, height - 1))
    header['displayWindow'] = Imath.Box2i(Imath.V2i(0, 0), Imath.V2i(width - 1, height - 1))
    header['lineOrder'] = Imath.LineOrder(Imath.LineOrder.INCREASING_Y)
    header['pixelAspectRatio'] = 1.0
    header['screenWindowCenter'] = (0.0, 0.0)
    header['screenWindowWidth'] = 1.0

    out = OpenEXR.OutputFile(file_path, header) # 지정된 파일 경로와 헤더를 사용하여 openexr 파일 생성
    print(header)

    depth_img_bytes = depth_img.tobytes()
    out.writePixels({'R': depth_img_bytes, 'G': depth_img_bytes, 'B': depth_img_bytes})
    out.close()