import pycolmap
import torch
import torch.nn as nn

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

    def from_pcd(self, pcd: PointCloud):
        self._size = pcd.size
        self._device = pcd.device

        coords = torch.tensor(pcd.coords, dtype=torch.float32, device=self._device)
        opacity = torch.tensor(pcd.colors[:, 3], dtype=torch.float32, device=self._device)
        sh = torch.zeros((self._size, 3, (self._sh_degree + 1) ** 2), dtype=torch.float32, device=self._device)
        sh[:, :, 0] = RGB2SH(pcd.colors[:, 3])

        scale = torch.ones((self._size, 3), device=self._device)
        rotation = torch.zeros((self._size, 4), device=self._device)
        rotation[:, 0] = 1

        self._coords = nn.Parameter(coords, requires_grad=True)
        self._opacity = nn.Parameter(opacity, requires_grad=True)
        self._sh = nn.Parameter(sh, requires_grad=True)
        self._scale = nn.Parameter(scale, requires_grad=True)
        self._rotation = nn.Parameter(rotation, requires_grad=True)

    def get_covariance(self, mask: torch.tensor = None):
        rot = self._rotation
        scale = self._scale
        if mask is not None:
            rot = rot[mask]
            scale = scale[mask]
        rot_matrices = Transform.get_rotation_matrices(rot, is_cuda=True)
        scale_matrices = Transform.get_scale_matrices(scale, is_cuda=True)
        ans = torch.bmm(rot_matrices, scale_matrices)
        return torch.bmm(ans, ans.transpose(1,2))
