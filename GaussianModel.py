import torch
import torch.nn as nn
import numpy as np
from plyfile import PlyData, PlyElement

# This magically reduce the memory load
from simple_knn._C import distCUDA2

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

        self.scale_activation = torch.exp

    @property
    def coords(self):
        return self._coords

    @property
    def covariance(self):
        return self.get_covariance()

    @property
    def scale(self):
        return self.scale_activation(self._scale)

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
        opacity = torch.ones(model._size, 1, dtype=torch.float32, device=model._device)
        sh = torch.zeros((model._size, 3, (model._sh_degree + 1) ** 2), dtype=torch.float32, device=model._device)
        sh[:, :, 0] = RGB2SH(torch.from_numpy(pcd.colors[:, :3]) / 255)

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.coords)).float().cuda()), 0.0000001)
        scale = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rotation = torch.zeros((model._size, 4), device=model._device)
        rotation[:, 0] = 1

        model._coords = nn.Parameter(coords, requires_grad=True)
        model._opacity = nn.Parameter(opacity, requires_grad=True)
        model._sh = nn.Parameter(sh, requires_grad=True)
        model._scale = nn.Parameter(scale, requires_grad=True)
        model._rotation = nn.Parameter(rotation, requires_grad=True)

        print(f"constructed {pcd.size} gaussians")
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

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._sh.shape[1] * self._sh.shape[2]):
            l.append('sh_{}'.format(i))
        l.append('opacity')
        for i in range(self._scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path: str):
        coords = self._coords.detach().cpu().numpy()
        normals = np.zeros_like(coords)
        sh = self._sh.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        opacity = self._opacity.detach().cpu().numpy()
        scale = self._scale.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(coords.shape[0], dtype=dtype_full)
        attributes = np.concatenate((coords, normals, sh, opacity, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path: str):
        plydata = PlyData.read(path)
        coords = np.stack((np.asarray(plydata.elements[0]["x"]),
                           np.asarray(plydata.elements[0]["y"]),
                           np.asarray(plydata.elements[0]["z"])), axis=1)
        opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        sh_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sh_")]
        sh_names = sorted(sh_names, key=lambda x: int(x.split('_')[-1]))
        assert len(sh_names) == 3 * (self.sh_degree + 1) ** 2
        sh = np.zeros((coords.shape[0], len(sh_names)))
        for idx, attr_name in enumerate(sh_names):
            sh[:, idx] = np.asarray(plydata.elements[0][attr_name])
        sh = sh.reshape((sh.shape[0], 3, (self.sh_degree + 1) ** 2))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scale = np.zeros((coords.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scale[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rot = np.zeros((coords.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rot[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._coords = nn.Parameter(torch.tensor(coords, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sh = nn.Parameter(torch.tensor(sh, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacity, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scale = nn.Parameter(torch.tensor(scale, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rot, dtype=torch.float, device="cuda").requires_grad_(True))
