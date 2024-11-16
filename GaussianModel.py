import torch
import torch.nn as nn
import numpy as np

from Utils.SphericalHarmonic import RGB2SH
from Utils.PointCloud import PointCloud
from Utils.Transform import Transform


class GaussianModel(nn.Module):
    def __init__(self, sh_degree: int = 3):
        super(GaussianModel, self).__init__()
        self._sh_degree = sh_degree
        self._coords = torch.empty(0)
        self._scale = torch.empty(0)
        self._rotation = torch.empty(0)
        self._sh = torch.empty(0)
        self._opacity = torch.empty(0)
        self._size = 0
        self._device = None

    @property
    def coords(self):
        return self._coords

    @property
    def scale(self):
        return self._scale

    @property
    def rotation(self):
        return self._rotation

    @property
    def opacity(self):
        return self._opacity

    @property
    def sh(self):
        return self._sh

    @property
    def sh_degree(self):
        return self._sh_degree

    @staticmethod
    def from_pcd(pcd: PointCloud) -> "GaussianModel":
        model = GaussianModel()
        model._size = pcd.size
        model._device = pcd.device

        coords = pcd.coords_torch
        opacity = torch.ones(model._size, dtype=torch.float32, device=model._device)
        sh = torch.zeros((model._size, 3, (model._sh_degree + 1) ** 2), dtype=torch.float32, device=model._device)
        sh[:, :, 0] = RGB2SH(torch.from_numpy(pcd.colors[:, :3]) / 255)

        scale = torch.ones((model._size, 3), device=model._device)
        rotation = torch.zeros((model._size, 4), device=model._device)
        rotation[:, 0] = 1

        model._coords = nn.Parameter(coords, requires_grad=True)
        model._opacity = nn.Parameter(opacity, requires_grad=True)
        model._sh = nn.Parameter(sh, requires_grad=True)
        model._scale = nn.Parameter(scale, requires_grad=True)
        model._rotation = nn.Parameter(rotation, requires_grad=True)
        return model

    def get_covariance(self, mask: torch.tensor = None):
        rot = self._rotation
        scale = self._scale
        if mask is not None:
            rot = rot[mask]
            scale = scale[mask]
        rot_matrices = Transform.get_rotation_matrices(rot, use_torch=True)
        scale_matrices = Transform.get_scale_matrices(scale, use_torch=True)
        ans = torch.bmm(rot_matrices, scale_matrices)
        return torch.bmm(ans, ans.transpose(1, 2))[:, :3, :3]

    def save_ply(self, path):
        from plyfile import PlyData, PlyElement
        coords = self._coords.detach().cpu().numpy()
        normals = np.zeros_like(coords)
        sh = self._sh.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacity = self._opacity.detach().cpu().numpy()
        scale = self._scale.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(coords.shape[0], dtype=dtype_full)
        attributes = np.concatenate((coords, normals, sh, opacity, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
