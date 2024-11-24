import torch
import torch.nn as nn
import numpy as np

from math import tan, atan
from Utils.Transform import Transform


class Camera(nn.Module):
    def __init__(self, position: np.ndarray, rotation: np.ndarray,
                 focal_length_x: float, focal_length_y: float,
                 principal_point_x: float, principal_point_y: float,
                 width: float, height: float):
        super(Camera, self).__init__()

        self.transform = Transform(position, rotation=rotation)
        self.fx = focal_length_x
        self.fy = focal_length_y
        self.cx = principal_point_x
        self.cy = principal_point_y
        self.height = height
        self.width = width

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.world2model_matrix = (torch.from_numpy(self.transform.get_world2model_matrix())
                                   .to(device=self.device, dtype=torch.float32))
        self.projection_matrix = (torch.from_numpy(self.get_projection_matrix())
                                  .to(device=self.device, dtype=torch.float32))

    @staticmethod
    def from_sfm(sfm_image: "pycolmap.Image", sfm_camera: "pycolmap.Camera", width: float, height: float) -> "Camera":
        # print("Start build Camera from SfM data")
        rigid3d = sfm_image.cam_from_world
        position = rigid3d.translation
        rotation = np.roll(rigid3d.rotation.quat, -1)    # has quaternion [x,y,z,w], need to change to [w,x,y,z]

        # print(sfm_camera.params_to_string)
        camera = Camera(position, rotation,
                        sfm_camera.focal_length_x, sfm_camera.focal_length_y,
                        sfm_camera.principal_point_x, sfm_camera.principal_point_y, width, height)
        # print(f"Success build Camera: pos={position}, rotation={rotation}, "
        #       f"width={width}, height={height}")
        return camera

    def get_projection_matrix(self, near: float = 0.1, far: float = 1000) -> np.ndarray:
        proj_matrix = np.array([
            [2 * self.fx / self.width, 0, -(2 * self.cx / self.width - 1), 0],
            [0, 2 * self.fy / self.height, -(2 * self.cy / self.height - 1), 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0],
        ], dtype=float)
        return proj_matrix

    def get_covariance2d(self, pos2d: torch.tensor, cov3d: torch.tensor):
        """
        project covariance matrix in 3d to 2d
        :param pos2d: gauss position in 3d (shape Nx3)
        :param cov3d: gauss covariance matrix in 3d (shape Nx3x3)
        :return: covariance matrix in 2d (shape Nx2x2)
        """
        W = self.world2model_matrix[:3, :3].T.to(self.device)
        J = torch.zeros((pos2d.shape[0], 2, 3), dtype=torch.float32, device=self.device)
        J[:, 0, 0] = 1 / pos2d[:, 2]
        J[:, 0, 2] = -pos2d[:, 0] / pos2d[:, 2]
        J[:, 1, 1] = 1 / pos2d[:, 2]
        J[:, 1, 2] = -pos2d[:, 1] / pos2d[:, 2]
        return J @ W @ cov3d @ W.T @ J.permute((0, 2, 1))

    def project_gaussian(self, screen_coords: torch.tensor, pos3d: torch.tensor, cov3d: torch.tensor, relax_factor: float = 1.3) -> (torch.tensor, torch.tensor):
        """
        Project gaussian sphere from 3d space to 2d
        :param pos3d: gaussian spheres' position (shape Nx3)
        :param cov3d: gaussian spheres' covariance matrix in 3d with shape (shape Nx3x3)
        :param relax_factor: tolerance range to accept gaussian sphere
        :return: position and covariance matrix on screen in 2d shape (shape Nx3, N, Nx2x2),
                 third element of position is its depth
        """

        ''' Project position into ndc space with MVP matrix '''
        # modelpos3d = pos3d_homo @ w2m
        pos3d_homo = torch.cat([pos3d, torch.ones((pos3d.shape[0], 1), device=self.device)], dim=1)
        p_ndc = pos3d_homo @ self.world2model_matrix.T @ self.projection_matrix.T
        p_ndc = p_ndc[:, :3] / torch.clamp(p_ndc[:, 3][:, None], min=0.000001)

        ''' Create cull mask, recall ndc space should be [-1, 1] '''
        mask = (0.2 <= p_ndc[:, 2]) & (p_ndc[:, 2] <= relax_factor) & \
               (-relax_factor <= p_ndc[:, 0]) & (p_ndc[:, 0] <= relax_factor) & \
               (-relax_factor <= p_ndc[:, 1]) & (p_ndc[:, 1] <= relax_factor)

        ''' Compute position and covariance in screen space with cull mask '''
        pos2d = screen_coords[mask] + p_ndc[mask]
        if pos2d.shape[0] <= 0:
            return None, None, mask
        pos2d[:, 0] = 0.5 * (pos2d[:, 0] + 1) * self.width
        pos2d[:, 1] = (1 - 0.5 * (pos2d[:, 1] + 1)) * self.height
        cov2d = self.get_covariance2d(pos3d_homo[mask], cov3d[mask])
        return pos2d, cov2d, mask
