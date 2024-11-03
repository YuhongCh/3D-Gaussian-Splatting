import torch
import torch.nn as nn

from GaussianModel import GaussianModel
from Utils.Camera import Camera
from Utils.Screen import Screen
from Utils.SphericalHarmonic import eval_sh
from Utils.Probability import gaussian_distribution


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
        return 3 * torch.sqrt(torch.maximum(term1 - term2, term1 + term2))

    def render(self, cam: Camera, num_tile: int = 16):
        device = self.model.coords.device

        # cull gaussian and project onto screen
        cov3d = self.model.get_covariance()
        pos2d, cov2d, mask = cam.project_gaussian(self.model.coords, cov3d)
        opacity = self.model.opacity[mask]
        color = eval_sh(self.model.sh_degree, self.model.sh[mask], self.model.coords[mask] - cam.transform.position)

        # compute radius of each gaussian sphere, which is determined by its largest eigenvalue
        radius = self.get_radius(cov2d)

        # create tiles and sort with prefix approach
        screen = Screen(cam.width, cam.height, num_tile, device=device)
        screen.create_tiles(pos2d, radius)
        screen.depth_sort(pos2d[:, 2])

        # parallelize to render
        render_color = torch.zeros((cam.width, cam.height), dtype=torch.float32, device=device)
        # render_alpha = torch.zeros((cam.width, cam.height), dtype=torch.float32, device=device)
        # render_depth = torch.zeros((cam.width, cam.height), dtype=torch.float32, device=device)
        streams = [torch.cuda.Stream(device=self.device) for _ in range(self.num_tile * self.num_tile)]
        for tid, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                left, top = screen.get_tl(tid)
                right, bottom = screen.get_br(tid)
                tis, tie = screen.get_indices_range(tid)

                # opacity[None] has shape Nx1 and gaussian distribution has shape NxM => shape NxM
                gauss_prob = gaussian_distribution(pos2d[tis:tie, :2], cov2d[tis:tie],
                                                   screen.pixel_pos[left:right, top:bottom])
                alpha = torch.clamp(opacity[None] * gauss_prob, min=0.01, max=0.99).permute((1,0)) # MxN
                weight = 1 - torch.cat([torch.zeros(alpha.shape[0]), alpha[:, :-1]], dim=0).cumsum(dim=1)
                weight = alpha * weight     # shape MxN

                tile_color = weight @ color[tis:tie]
                render_color[left:right, top:bottom] = tile_color.reshape((right - left, bottom - top))

                # tile_alpha = weight.sum(dim=1)
                # render_color[left:right, top:bottom] = tile_color.reshape((right - left, bottom - top))
                #
                # tile_depth =
                # render_color[left:right, top:bottom] = tile_color.reshape((right - left, bottom - top))
                # render
        return render_color
