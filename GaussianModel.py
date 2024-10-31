import pycolmap
import torch
import torch.nn as nn

from Utils.SphericalHarmonic import rgb2sh
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

    def from_pcd(self, pcd: PointCloud):
        coords = torch.tensor(pcd.coords).cuda()
        opacity = torch.tensor(pcd.colors[:, 3]).cuda()
        sh = torch.tensor(rgb2sh(pcd.colors[:, :3])).cuda()
        scale = torch.ones((pcd.size, 3), device="cuda")
        rotation = torch.zeros((pcd.size, 4), device="cuda")
        rotation[:, 0] = 1

        self._size = pcd.size
        self._coords = nn.Parameter(coords, requires_grad=True)
        self._opacity = nn.Parameter(opacity, requires_grad=True)
        self._sh = nn.Parameter(sh, requires_grad=True)
        self._scale = nn.Parameter(scale, requires_grad=True)
        self._rotation = nn.Parameter(rotation, requires_grad=True)

    def get_covariance(self):
        rot_matrices = Transform.get_rotation_matrices(self._rotation, is_cuda=True)
        scale_matrices = Transform.get_scale_matrices(self._scale, is_cuda=True)
        ans = torch.bmm(rot_matrices, scale_matrices)
        return torch.bmm(ans, ans.transpose(1,2))
    