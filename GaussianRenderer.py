import torch
import torch.nn as nn

from GaussianModel import GaussianModel
from Utils.Camera import Camera
from Utils.Screen import Screen
from Utils.SphericalHarmonic import eval_sh
from Utils.Probability import gaussian_distribution


class GaussianRenderer(nn.Module):
    def __init__(self, model: GaussianModel):
        super(GaussianRenderer, self).__init__()
        self.model = model
        self.device = self.model.coords.device
        self.background_color = torch.tensor([0, 0, 0])

    @staticmethod
    def get_radius(cov2d: torch.tensor) -> torch.tensor:
        trace = cov2d[:, 0, 0] + cov2d[:, 1, 1]
        det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
        term1 = 0.5 * trace
        term2 = 0.5 * torch.sqrt(torch.clamp(trace * trace - 4 * det, min=0))
        return 3 * torch.sqrt(torch.maximum(term1 - term2, term1 + term2))

    def get_empty_image(self, cam: Camera):
        return torch.zeros((cam.width, cam.height, 3), dtype=torch.float, device=self.device)

    def render(self, cam: Camera, num_tile: int = 16):
        # cull gaussian and project onto screen
        cov3d = self.model.get_covariance()
        pos2d, cov2d, mask = cam.project_gaussian(self.model.coords, cov3d)
        if pos2d is None and cov2d is None:
            return self.get_empty_image(cam)
        opacity = self.model.opacity[mask]
        color = eval_sh(self.model.sh_degree,
                        self.model.sh[mask].permute(0,2,1),
                        self.model.coords[mask] - torch.tensor(cam.transform.position, dtype=torch.float32, device=self.device))
        color = torch.clamp(color + 0.5, min=0.0)

        # compute radius of each gaussian sphere, which is determined by its largest eigenvalue
        radius = self.get_radius(cov2d)

        # create tiles and sort with prefix approach
        screen = Screen(cam.width, cam.height, num_tile, device=self.device)
        screen.create_tiles(pos2d, radius)
        screen.depth_sort(pos2d[:, 2])

        # parallelize to render
        render_color = torch.zeros((cam.width, cam.height, 3), dtype=torch.float32, device=self.device)
        # render_alpha = torch.zeros((cam.width, cam.height), dtype=torch.float32, device=device)
        # render_depth = torch.zeros((cam.width, cam.height), dtype=torch.float32, device=device)
        streams = [torch.cuda.Stream(device=self.device) for _ in range(num_tile * num_tile)]
        for tid, stream in enumerate(streams):
            print(f"Compute tid {tid}")
            with torch.cuda.stream(stream):
                left, top = screen.get_tl(tid)
                right, bottom = screen.get_br(tid)
                tis, tie = screen.get_indices_range(tid)
                gauss_indices = screen.tile_indices[tis:tie]

                # if does not find a gauss in the region, continue
                curr_pos2d = pos2d[gauss_indices, :2]
                curr_cov2d = cov2d[gauss_indices]
                curr_opacity = opacity[gauss_indices]
                curr_color = color[gauss_indices]
                if curr_pos2d.shape[0] == 0:
                    continue

                # opacity[None] has shape Nx1 and gaussian distribution has shape NxM1xM2x2 => shape NxM1xM2
                gauss_prob = gaussian_distribution(curr_pos2d, curr_cov2d,
                                                   screen.pixel_pos[left:right, top:bottom])
                # alpha has shape M1xM2xN
                alpha = torch.clamp(curr_opacity.view(-1, 1, 1) * gauss_prob,
                                    min=0.01, max=0.99).permute((1, 2, 0))
                weight = torch.cat([
                    torch.ones((alpha.shape[0], alpha.shape[1], 1), device=self.device),
                    1 - alpha[:, :, :-1]
                ], dim=2).cumprod(dim=2)
                assert((weight < 0).sum() == 0)

                # (M1xM2xN)x(Nx3) => (M1xM2x1xN)x(Nx3) => M1xM2x1x3 => M1xM2x3
                tile_color = (alpha * weight).unsqueeze(2) @ curr_color
                render_color[left:right, top:bottom] = torch.squeeze(tile_color)

                # tile_alpha = weight.sum(dim=1)
                # render_color[left:right, top:bottom] = tile_color.reshape((right - left, bottom - top))
                #
                # tile_depth =
                # render_color[left:right, top:bottom] = tile_color.reshape((right - left, bottom - top))
                # render
        return render_color
