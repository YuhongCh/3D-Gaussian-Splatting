import torch
import torch.nn as nn

from numpy import clip
from math import ceil


class Screen(nn.Module):
    def __init__(self, width: int, height: int, num_tile: int = 16, device: str = None):
        super(Screen, self).__init__()
        self.width = width
        self.height = height
        self.num_tile = num_tile

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.block_width = ceil(self.height / self.num_tile)
        self.block_height = ceil(self.height / self.num_tile)

        self.pixel_pos = torch.stack(
            torch.meshgrid(torch.arange(self.width), torch.arange(self.height), indexing='xy'), dim=-1
        ).to(self.device)
        self.tile_count = torch.zeros(self.num_tile * self.num_tile, dtype=torch.int32, device=self.device)
        self.tile_indices = torch.empty(0)

    def create_tiles(self, pos2d: torch.tensor, radius: torch.tensor):
        minval = 0
        maxval = self.num_tile - 0.00001
        xmin_indices = torch.clamp(pos2d[:, 0] / self.block_width - radius, min=minval, max=maxval).to(torch.int32)
        xmax_indices = torch.clamp(pos2d[:, 0] / self.block_width + radius, min=minval, max=maxval).to(torch.int32)
        ymin_indices = torch.clamp(pos2d[:, 1] / self.block_height - radius, min=minval, max=maxval).to(torch.int32)
        ymax_indices = torch.clamp(pos2d[:, 1] / self.block_height + radius, min=minval, max=maxval).to(torch.int32)

        for index in range(pos2d.shape[0]):
            for xi in range(xmin_indices[index], xmax_indices[index] + 1):
                for yi in range(ymin_indices[index], ymax_indices[index] + 1):
                    self.tile_count[xi + yi * self.num_tile] += 1
        self.tile_count = torch.cumsum(self.tile_count, dim=0, dtype=torch.int32)
        count = torch.zeros(self.tile_count.shape[0], dtype=torch.int32, device=self.device)

        self.tile_indices = torch.zeros(self.tile_count[-1], dtype=torch.int32, device=self.device)
        for index in range(pos2d.shape[0]):
            for xi in range(xmin_indices[index], xmax_indices[index] + 1):
                for yi in range(ymin_indices[index], ymax_indices[index] + 1):
                    tile_index = self.convert_2dto1d(xi, yi)
                    prev = 0 if tile_index == 0 else self.tile_count[tile_index - 1]
                    target_index = prev + count[tile_index]
                    self.tile_indices[target_index] = index
                    count[tile_index] += 1

    def depth_sort(self, depth: torch.tensor):
        tile_streams = [torch.cuda.Stream(device=self.device) for _ in range(self.num_tile * self.num_tile)]
        for si, stream in enumerate(tile_streams):
            with torch.cuda.stream(stream):
                prev, curr = self.get_indices_range(si)
                sorted_indices = torch.argsort(depth[self.tile_indices[prev:curr]])
                self.tile_indices[prev:curr] = self.tile_indices[prev:curr][sorted_indices]

    def convert_2dto1d(self, xi: int, yi: int):
        return xi + self.num_tile * yi

    def convert_1dto2d(self, ti: int):
        yi = ti // self.num_tile
        xi = ti - yi * self.num_tile
        return xi, yi

    def get_tl(self, tile_index: int) -> (int, int):
        assert (0 <= tile_index < self.num_tile * self.num_tile)
        xi, yi = self.convert_1dto2d(tile_index)
        return xi * self.block_width, yi * self.block_height

    def get_br(self, tile_index: int) -> (int, int):
        assert (0 <= tile_index < self.num_tile * self.num_tile)
        xi, yi = self.convert_1dto2d(tile_index)
        right = clip((xi + 1) * self.block_width, 0, self.width, dtype=int)
        bottom = clip((yi + 1) * self.block_height, 0, self.height, dtype=int)
        return right, bottom

    def get_indices_range(self, tile_index: int) -> (int, int):
        assert (0 <= tile_index < self.num_tile * self.num_tile)
        prev = 0 if tile_index == 0 else self.tile_count[tile_index - 1]
        return prev, self.tile_count[tile_index]
