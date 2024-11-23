import torch
import torch.nn as nn


from GaussianModel import GaussianModel
from Utils.Camera import Camera
from Utils.Screen import Screen
from Utils.SphericalHarmonic import eval_sh
from Utils.Probability import gaussian_distribution


class GaussianRenderer(nn.Module):
    def __init__(self, model: GaussianModel, debug: bool = False):
        super(GaussianRenderer, self).__init__()
        self.model = model
        self.device = self.model.coords.device
        self.background_color = torch.tensor([0, 0, 0])

        if debug:
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
        else:
            self.prof = None

    @staticmethod
    def get_radius(cov2d: torch.tensor) -> torch.tensor:
        """get maxmimum radius, *3 so that we are in 99% confidence zone"""
        trace = cov2d[:, 0, 0] + cov2d[:, 1, 1]
        det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d [:, 0, 1] * cov2d[:, 1, 0]
        term1 = 0.5 * trace
        term2 = 0.5 * torch.sqrt(torch.clamp(trace * trace - 4 * det, min=0))
        return 3 * torch.sqrt(torch.maximum(term1 - term2, term1 + term2))

    def get_empty_image(self, cam: Camera):
        return torch.zeros((cam.width, cam.height, 3), dtype=torch.float, device=self.device)

    def forward(self, cam: Camera, num_tiles: int = 32) -> torch.Tensor:
        return self.render(cam, num_tiles)

    def render(self, cam: Camera, tile_length: int = 32):
        # cull gaussian and project onto screen
        pos2d, cov2d, mask = cam.project_gaussian(self.model.coords, self.model.covariance)
        if pos2d is None and cov2d is None:
            print(f"Observed 0 points")
            return self.get_empty_image(cam)
        print(f"Observed {pos2d.shape[0]} points")

        opacity = self.model.opacity[mask]
        color = eval_sh(self.model.sh_degree,
                        self.model.sh[mask],
                        self.model.coords[mask] - torch.tensor(cam.transform.position, dtype=torch.float32, device=self.device))
        color = torch.clamp(color + 0.5, min=0.0)

        # compute radius of each gaussian sphere, which is determined by its largest eigenvalue
        radius = self.get_radius(cov2d)

        # create tiles and sort with prefix approach
        screen = Screen(cam.width, cam.height, tile_length, device=self.device)
        screen.create_tiles(pos2d, radius)
        screen.depth_sort(pos2d[:, 2])

        # parallelize to render
        render_color = torch.zeros((cam.width, cam.height, 3), dtype=torch.float32, device=self.device)
        for tid in range(screen.num_block):
            # self.prof.start()
            left, top = screen.get_tl(tid)
            right, bottom = screen.get_br(tid)
            tis, tie = screen.get_indices_range(tid)
            gauss_indices = screen.tile_indices[tis:tie]
            # print(f"tid({tid}), LR({left}, {right}), TB({top}, {bottom}), TSE({tis}, {tie}), num gauss({gauss_indices.shape[0]})")

            # if does not find a gauss in the region, continue
            curr_pos2d = pos2d[gauss_indices, :2]
            curr_cov2d = cov2d[gauss_indices]
            curr_opacity = opacity[gauss_indices]
            curr_color = color[gauss_indices]
            if curr_pos2d.shape[0] == 0:
                continue

            #  Shape NxM1xM2
            gauss_prob = gaussian_distribution(curr_pos2d, curr_cov2d,
                                               torch.stack(
                                                   torch.meshgrid(torch.arange(left, right),
                                                                  torch.arange(top, bottom), indexing='ij'), dim=-1
                                               ).to(self.device),
                                               batch_size=64)

            # alpha has shape M1xM2xN
            alpha = torch.clamp(curr_opacity.view(-1, 1, 1) * gauss_prob,
                                min=0.01, max=0.99).permute((1, 2, 0))
            weight = torch.cat([
                torch.ones((alpha.shape[0], alpha.shape[1], 1), device=self.device),
                1 - alpha[:, :, :-1]
            ], dim=2).cumprod(dim=2)

            # (M1xM2xN)x(Nx3) => (M1xM2x1xN)x(Nx3) => M1xM2x1x3 => M1xM2x3
            render_color[left:right, top:bottom] += (alpha * weight) @ curr_color

        return render_color
