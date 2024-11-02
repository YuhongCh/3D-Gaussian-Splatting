import torch
import torch.nn as nn

from GaussianModel import GaussianModel
from Utils.Camera import Camera
from Utils.Transform import Transform


class GaussianRender(nn.Module):
    def __init__(self, model: GaussianModel):
        super(GaussianRender, self).__init__()
        self.model = model
        self.background_color = torch.tensor([0, 0, 0])

    @staticmethod
    def get_radius(cov2d: torch.tensor) -> torch.tensor:
        trace = cov2d[:, 0, 0] + cov2d[:, 1, 1]
        det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
        term1 = 0.5 * trace
        term2 = 0.5 * torch.sqrt(torch.clamp(torch.mintrace * trace - 4 * det, min=0))
        return torch.maximum(term1 - term2, term1 + term2)

    def render(self, cam: Camera, num_tile: int = 16):
        # cull gaussian
        device = self.model.coords.device
        mask = cam.get_cull_mask(self.model.coords)

        # project onto screen
        cov3d = self.model.get_covariance(mask)
        pos3d = self.model.coords[mask]
        opacity = self.model.opacity[mask]
        pos2d, cov2d = cam.project_gaussian(pos3d, cov3d)

        # compute radius of each gaussian sphere, which is determined by its largest eigenvalue
        radius = self.get_radius(cov2d)

        # create tiles and sort with prefix approach
        tile_count = torch.zeros((num_tile, num_tile), dtype=torch.int32, device=device)
            # use AABB approach to roughly determine the correspondent grid
        for pindex in range(pos2d.shape[0]):
            pos2d[jk]
        tile_count = torch.cumsum(tile_count, dim=0, dtype=torch.int32)
        tile_indices = torch.zeros(tile_count[-1], dtype=torch.int32, device=device)

        # tile_streams = [torch.cuda.Stream(device=device) for _ in range(num_tile * num_tile)]
        # for stream in tile_streams:
        #     with torch.cuda.stream(stream):
        #         for







