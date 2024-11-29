import torch
import torch.nn as nn

from numpy import clip
from math import ceil


class Screen(nn.Module):
    def __init__(self, width: int, height: int, tile_length: int = 16, device: str = None):
        super(Screen, self).__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.width = width
        self.height = height
        self.tile_length = tile_length

        self.xmin = torch.empty(0)
        self.xmax = torch.empty(0)
        self.ymin = torch.empty(0)
        self.ymax = torch.empty(0)

        self.num_points = None
        self.num_block_width = int(ceil(self.width / self.tile_length))
        self.num_block_height = int(ceil(self.height / self.tile_length))
        self.num_block = self.num_block_height * self.num_block_width

    def create_tiles(self, pos2d: torch.tensor, radius: torch.tensor):
        self.num_points = pos2d.shape[0]
        # print(f"start create tiles for {self.num_points} points")
        self.xmin = torch.clamp(pos2d[:, 0] - radius, min=0, max=self.width).to(torch.int32)
        self.ymin = torch.clamp(pos2d[:, 1] - radius, min=0, max=self.height).to(torch.int32)
        self.xmax = torch.clamp(pos2d[:, 0] + radius, min=0, max=self.width).to(torch.int32)
        self.ymax = torch.clamp(pos2d[:, 1] + radius, min=0, max=self.height).to(torch.int32)

    def convert_2dto1d(self, xi: int, yi: int):
        return yi + self.num_block_height * xi

    def convert_1dto2d(self, ti: int):
        xi = ti // self.num_block_height
        yi = ti - xi * self.num_block_height
        return xi, yi

    def get_tl(self, tile_index: int) -> (int, int):
        assert (0 <= tile_index < self.num_block)
        xi, yi = self.convert_1dto2d(tile_index)
        return xi * self.tile_length, yi * self.tile_length

    def get_br(self, tile_index: int) -> (int, int):
        assert (0 <= tile_index < self.num_block)
        xi, yi = self.convert_1dto2d(tile_index)
        right = clip((xi + 1) * self.tile_length, 0, self.width, dtype=int)
        bottom = clip((yi + 1) * self.tile_length, 0, self.height, dtype=int)
        return right, bottom

    def get_inmask(self, left, right, top, bottom) -> torch.Tensor:
        over_tl = self.xmin.clip(min=left), self.ymin.clip(min=top)
        over_br = self.xmax.clip(max=right), self.ymax.clip(max=bottom)
        in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
        return in_mask
