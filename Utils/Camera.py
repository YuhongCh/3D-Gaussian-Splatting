import pycolmap
import torch
import numpy as np
import torch.nn as nn
import Utils.Transform as Transform

from math import tan, atan


class Camera(nn.Module):
    def __init__(self, position, rotation, fovX, fovY, target_image, znear=0.1, zfar=100):
        super(Camera, self).__init__()

        self.transform = Transform(position, rotation=rotation)
        self.fovX = fovX
        self.fovY = fovY
        self.znear = znear
        self.zfar = zfar

        self.target_image = target_image
        self.height = self.target_image.shape[0]
        self.width = self.target_image.shape[1]

        self.camera2world_matrix = self.transform.get_model2world_matrix()
        self.word2camera_matrix = self.transform.get_world2model_matrix()
        self.projection_matrix = self.get_projection_matrix(self.znear, self.zfar, self.fovX, self.fovY)

    def get_covariance2d(self, ppos: torch.tensor, pcov: torch.tensor):
        """
        project covariance matrix in 3d to 2d
        :param ppos: gauss position in 3d (shape Nx3)
        :param pcov: gauss covariance matrix in 3d (shape Nx3x3)
        :return: covariance matrix in 2d (shape Nx2x2)
        """
        W = self.transform.get_world2model_matrix()[:3, :3].transpose()
        J = torch.zeros((ppos.shape[0], 2, 3), dtype=torch.float32, device=ppos.device)
        J[:, 0, 0] = 1 / ppos[:, 2]
        J[:, 0, 2] = -ppos[:, 0] / ppos[:, 2]
        J[:, 1, 1] = 1 / ppos[:, 2]
        J[:, 1, 2] = -ppos[:, 1] / ppos[:, 2]
        return J @ W @ pcov @ W.transpose() @ J.permute((0, 2, 1))

    def get_cull_mask(self, ppos: torch.tensor, relax_factor: torch.float32 = 1.3) -> torch.tensor:
        """
        Compute the cull mask of ppos to indicate which points should be removed in rendering
        :param ppos: point position (shape Nx3)
        :param relax_factor: expand the view frustum to contain more gaussian spheres
        :return: mask (shape N)
        """
        view_matrix = self.transform.get_world2model_matrix().transpose()
        p_modelpos = ppos @ view_matrix[:3, :3] + view_matrix[3, :3]
        relax_factor = 1.3
        x_ratio = tan(0.5 * self.fovX)
        y_ratio = tan(0.5 * self.fovY)
        mask = ((relax_factor * -x_ratio <= p_modelpos[:, 0] / p_modelpos[:, 2]) &
                (p_modelpos[:, 0] / p_modelpos[:, 2] <= relax_factor * x_ratio) &
                (relax_factor * -y_ratio <= p_modelpos[:, 1] / p_modelpos[:, 2]) &
                (p_modelpos[:, 1] / p_modelpos[:, 2] <= relax_factor * y_ratio))
        return mask

    def project_gaussian(self, pos3d: torch.tensor, cov3d: torch.tensor, relax_factor: float = 1.3) -> (torch.tensor, torch.tensor):
        """
        Project gaussian sphere from 3d space to 2d
        :param ppos: gaussian spheres' position (shape Nx3)
        :param pcov: gaussian spheres' covariance matrix in 3d with shape (shape Nx3x3)
        :return: position and covariance matrix on screen in 2d shape (shape Nx3, N, Nx2x2),
                 third element of position is its depth
        """

        ''' Project position into ndc space with MVP matrix '''
        device = pos3d.device
        view_matrix = self.transform.get_world2model_matrix().transpose()
        projection_matrix = self.get_projection_matrix(self.znear, self.zfar, self.fovX, self.fovY).transpose()
        p_modelpos = pos3d @ view_matrix[:3, :3] + view_matrix[3, :3]
        p_modelpos = torch.cat([p_modelpos, torch.ones((p_modelpos.shape[0], 1), device=p_modelpos.device)], dim=1)
        p_ndc = p_modelpos @ projection_matrix
        p_ndc = p_ndc[:, :3] / (p_ndc[:, 3].unsqueeze(1) + 0.000001)

        ''' Create cull mask, recall ndc space should be [-1, 1] '''
        mask = (p_ndc[:, 2] >= 0.2 &
                -relax_factor <= p_ndc[:, 0] & p_ndc[:, 0] <= relax_factor &
                -relax_factor <= p_ndc[:, 1] & p_ndc[:, 1] <= relax_factor)

        ''' Compute position and covariance in screen space with cull mask '''
        pos2d = p_ndc[mask]
        pos2d[:, 0] = 0.5 * (pos2d[:, 0] + 1) * self.width
        pos2d[:, 1] = (1 - 0.5 * (pos2d[:, 1] + 1)) * self.height
        cov2d = self.get_covariance2d(pos3d[mask], cov3d[mask], device=device)
        return pos2d, cov2d, mask

    @staticmethod
    def get_projection_matrix(znear: float, zfar: float, fovX: float, fovY: float):
        """
        :param znear: screen distance to render the image
        :param zfar: maximum distance to have object visible on screen
        :param fjwovX: field of view in X axis, represent width
        :param fovY: field of view in Y axis, represent height
        :return: projection matrix to project points in view frustum onto screen image
        """
        right = znear * tan(0.5 * fovX)
        top = znear * tan(0.5 * fovY)
        left = -right
        bottom = -top
        return np.array([
            [2 * znear / (right - left), 0, (left + right) / (right - left), 0],
            [0, 2 * znear / (top - bottom), (top + bottom) / (top - bottom), 0],
            [0, 0, zfar / (zfar - znear), -zfar * znear / (zfar - znear)],
            [0, 0, 1, 0]
        ], dtype=float)

    @staticmethod
    def focal2fov(focal, pixels):
        """
        :param focal: focal length of camera
        :param pixels: height or width of camera screen in terms of number of pixels
        :return: fov of camera
        """
        return 2 * atan(pixels / (2 * focal))

    @staticmethod
    def fov2focal(fov, pixels):
        """
        similar as focal2fov
        """
        return pixels / (2 * tan(fov / 2))
