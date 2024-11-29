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
    def __init__(self, max_sh_degree: int = 3):
        super(GaussianModel, self).__init__()
        self.max_sh_degree = max_sh_degree
        self.curr_sh_degree = 0
        self._coords = torch.empty(0)
        self._scale = torch.empty(0)
        self._rotation = torch.empty(0)
        self._sh = torch.empty(0)
        self._opacity = torch.empty(0)
        self._coords_grads = torch.empty(0)
        self._denom = torch.empty(0)
        self._size = 0
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self.opactiy_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.scale_activation = torch.exp
        self.scale_inv_activation = torch.log

    @property
    def size(self):
        return self._size

    @property
    def capacity(self):
        return self._coords.shape[0]

    @property
    def coords(self):
        return self._coords[:self._size]

    @property
    def covariance(self):
        return self.get_covariance()

    @property
    def scale(self):
        return self.scale_activation(self._scale[:self._size])

    @property
    def rotation(self):
        return self.rotation_activation(self._rotation[:self._size])

    @property
    def opacity(self):
        return self.opactiy_activation(self._opacity[:self._size])

    @property
    def sh(self):
        num_params = (self.curr_sh_degree + 1) ** 2
        return self._sh[:self._size, :, :num_params]

    @property
    def sh_degree(self):
        return self.curr_sh_degree

    @staticmethod
    def from_pcd(pcd: PointCloud) -> "GaussianModel":
        model = GaussianModel()
        model._size = pcd.size
        model._device = pcd.device

        coords = pcd.coords_torch
        opacity = torch.ones(model._size, 1, dtype=torch.float32, device=model._device)
        sh = torch.zeros((model._size, 3, (model.max_sh_degree + 1) ** 2), dtype=torch.float32, device=model._device)
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

        model._coords_grads = torch.zeros((model.capacity, 1), dtype=torch.float32, device=model._device)
        model._denom = torch.zeros((model.capacity, 1), dtype=torch.int32, device=model._device)

        print(f"constructed {pcd.size} gaussians")
        return model

    def get_covariance(self, mask: torch.tensor = None):
        rot = self.rotation
        scale = self.scale
        if mask is not None:
            rot = rot[mask]
            scale = scale[mask]
        rot_matrices = Transform.get_rotation_matrices(rot, use_torch=True)
        scale_matrices = Transform.get_scale_matrices(scale, use_torch=True)
        ans = torch.bmm(rot_matrices, scale_matrices)
        return torch.bmm(ans, ans.transpose(1, 2))[:, :3, :3]

    def reset_opacity(self):
        self._opacity[:, :] = 0.01

    def add_sh_degree(self):
        self.curr_sh_degree = min(self.max_sh_degree, 1 + self.curr_sh_degree)

    def expand_capacity(self, new_capacity: int):
        if self.capacity >= new_capacity:
            return

        # gaussian sphere properties
        capacity_diff = new_capacity - self.capacity
        self._coords = nn.Parameter(torch.cat(
            (self._coords, torch.zeros((capacity_diff, 3), dtype=torch.float32, device=self._device)), dim=0
        )).requires_grad_(True)
        self._opacity = nn.Parameter(torch.cat(
            (self._opacity, torch.zeros((capacity_diff, 1), dtype=torch.float32, device=self._device)), dim=0
        )).requires_grad_(True)
        self._sh = nn.Parameter(torch.cat(
            (self._sh, torch.zeros((capacity_diff, 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float32, device=self._device)), dim=0
        )).requires_grad_(True)
        self._scale = nn.Parameter(torch.cat(
            (self._scale, torch.zeros((capacity_diff, 3), dtype=torch.float32, device=self._device)), dim=0
        )).requires_grad_(True)
        self._rotation = nn.Parameter(torch.cat(
            (self._rotation, torch.zeros((capacity_diff, 4), dtype=torch.float32, device=self._device)), dim=0
        )).requires_grad_(True)

        # grad properties
        self._coords_grads = torch.cat((self._coords_grads, torch.zeros((capacity_diff, 1), dtype=torch.float32, device=self._device)), dim=0)
        self._denom = torch.cat((self._denom, torch.zeros((capacity_diff, 1), dtype=torch.int32, device=self._device)), dim=0)

    def remove(self, mask: torch.Tensor, inplace: bool = True):
        """
        Remove elements based on mask, if mask is True, remove it.
        :param mask: determine whether to remove the element
        :param inplace: if in place, remove without update the capacity
        """
        if inplace:
            num_removed = mask.sum()
            if num_removed == 0:
                return

            mask = ~mask
            start_index = self._size - num_removed - num_removed
            self._coords[start_index: start_index + num_removed] = self.coords[mask][-num_removed:]
            self._opacity[start_index: start_index + num_removed] = self.opacity[mask][-num_removed:]
            self._rotation[start_index: start_index + num_removed] = self.rotation[mask][-num_removed:]
            self._sh[start_index: start_index + num_removed] = self._sh[:self._size, :, :][mask][-num_removed:]
            self._scale[start_index: start_index + num_removed] = self.scale[:self._size][mask][-num_removed:]

            self._coords_grads[start_index: start_index + num_removed] = self._coords_grads[:self._size][mask][-num_removed:]
            self._denom[start_index: start_index + num_removed] = self._denom[:self._size][mask][-num_removed:]

            self._size -= num_removed
        else:
            # TODO: Check if there is Bug Later
            mask = ~mask
            self._coords = self.coords[mask]
            self._opacity = self.opacity[mask]
            self._sh = self._sh[mask]
            self._scale = self.opacity[mask]
            self._rotation = self.rotation[mask]

            self._coords_grads = self._coords_grads[:self._size][mask]
            self._denom = self._denom[:self._size][mask]

            self._size = self._coords.shape[0]

    def clone(self, mask: torch.Tensor, expand_factor: torch.uint8 = 2):
        num_clone = mask.sum()
        if num_clone == 0:
            return

        new_size = self._size + num_clone
        if new_size >= self.capacity:
            new_capacity = self._size + num_clone * expand_factor
            self.expand_capacity(new_capacity)
        self._coords[self._size: new_size] = self._coords[:self._size][mask]
        self._opacity[self._size: new_size] = self._opacity[:self._size][mask]
        self._rotation[self._size: new_size] = self._rotation[:self._size][mask]
        self._sh[self._size: new_size] = self._sh[:self._size][mask]
        self._scale[self._size: new_size] = self._scale[:self._size][mask]

        self._coords_grads = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self._device)
        self._denom = torch.zeros((self.capacity, 1), dtype=torch.int32, device=self._device)
        self._size = new_size

    def split(self, mask: torch.Tensor, expand_factor: torch.uint8 = 2, split_factor: torch.uint8 = 2):
        num_split = mask.sum()
        if num_split == 0:
            return

        new_size = self._size + num_split * (split_factor - 1)
        if new_size >= self.capacity:
            new_capacity = self._size + num_split * (split_factor - 1) * expand_factor
            self.expand_capacity(new_capacity)

        stds = self._scale[:self._size][mask].repeat(split_factor, 1)
        sample_means = torch.normal(torch.zeros((num_split * split_factor, 3), device=self._device), std=stds)
        rot_matrix = Transform.get_rotation_matrices(self._rotation[:self._size][mask], use_torch=True).repeat(split_factor, 1, 1).transpose(1, 2)
        sample_means = torch.squeeze(sample_means[:, None, :] @ rot_matrix[:, :3, :3]) + rot_matrix[:, 3, :3]
        sample_scales = self.scale_inv_activation(self._scale[:self._size][mask].repeat(split_factor, 1) / (0.8*split_factor))

        # assign to original position
        self._coords[:self._size][mask] = sample_means[:num_split]
        self._scale[:self._size][mask] = sample_scales[:num_split]

        # assign to new positions
        self._coords[self._size: new_size] = sample_means[num_split:]
        self._scale[self._size: new_size] = sample_scales[num_split:]
        self._rotation[self._size: new_size] = self._rotation[:self._size][mask].repeat(split_factor - 1, 1, 1)
        self._opacity[self._size: new_size] = self._opacity[:self._size][mask].repeat(split_factor - 1, 1)
        self._sh[self._size: new_size] = self._sh[:self._size][mask].repeat(split_factor - 1, 1, 1)

        self._coords_grads = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self._device)
        self._denom = torch.zeros((self.capacity, 1), dtype=torch.int32, device=self._device)
        self._size = new_size

    def update_densify_stats(self, pos2d: torch.Tensor, mask: torch.Tensor):
        self._coords_grads[:self._size][mask] += torch.norm(pos2d.grad[mask, :2], dim=-1, keepdim=True)
        self._denom[:self._size][mask] += 1

    def densify(self, scene_radius: torch.float32):
        # TODO: This might still need to limit the size a gaussian can display on screen
        grads = self._coords_grads / self._denom
        grads[grads.isnan()] = 0.0

        # determine clone and split
        clone_mask = (torch.norm(grads[:self._size], dim=-1) >= 0.0002) & (torch.max(self._scale[:self._size], dim=1).values <= 0.01 * scene_radius)
        self.clone(clone_mask)
        pad_grads = torch.zeros((self._size, 1), dtype=torch.float32, device=self._device)
        pad_grads[:min(self._size, grads.shape[0])] = grads[:self._size]
        split_mask = (torch.norm(pad_grads, dim=-1) >= 0.0002) & (torch.max(self._scale[:self._size], dim=1).values > 0.01 * scene_radius)
        self.split(split_mask[:self._size])

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
        self._size = coords.shape[0]
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        sh_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sh_")]
        sh_names = sorted(sh_names, key=lambda x: int(x.split('_')[-1]))
        assert len(sh_names) == 3 * (self.max_sh_degree + 1) ** 2
        sh = np.zeros((coords.shape[0], len(sh_names)))
        for idx, attr_name in enumerate(sh_names):
            sh[:, idx] = np.asarray(plydata.elements[0][attr_name])
        sh = sh.reshape((sh.shape[0], 3, (self.max_sh_degree + 1) ** 2))

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

        self._coords = nn.Parameter(torch.tensor(coords, dtype=torch.float, device=self._device).requires_grad_(True))
        self._sh = nn.Parameter(torch.tensor(sh, dtype=torch.float, device=self._device).requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacity, dtype=torch.float, device=self._device).requires_grad_(True))
        self._scale = nn.Parameter(torch.tensor(scale, dtype=torch.float, device=self._device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rot, dtype=torch.float, device=self._device).requires_grad_(True))
        self._coords_grads = torch.zeros_like(self._opacity, dtype=torch.float, device=self._device)
        self._denom = torch.zeros_like(self._opacity, dtype=torch.int, device=self._device)

    def capture(self, optimizer):
        return (
            self.curr_sh_degree,
            self._coords,
            self._sh,
            self._scale,
            self._rotation,
            self._opacity,
            self._coords_grads,
            self._denom,
            optimizer.state_dict(),
        )
