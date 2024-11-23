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

        self.num_points = None
        self.num_block_width = int(ceil(self.width / self.tile_length))
        self.num_block_height = int(ceil(self.height / self.tile_length))
        self.num_block = self.num_block_height * self.num_block_width

        self.tile_count = torch.empty(0)    # prefix sum start with 0
        self.tile_indices = torch.empty(0)

        # print(f"screen has shape(W, H) = ({self.width}, {self.height}), "
        #       f"with num block shape (W, H) = ({self.num_block_width}, {self.num_block_height})")

    def create_tiles(self, pos2d: torch.tensor, radius: torch.tensor):
        self.num_points = pos2d.shape[0]
        # print(f"start create tiles for {self.num_points} points")
        xmin = torch.clamp((pos2d[:, 0] - radius) // self.tile_length, min=0, max=self.num_block_width).to(torch.int32)
        ymin = torch.clamp((pos2d[:, 1] - radius) // self.tile_length, min=0, max=self.num_block_height).to(torch.int32)
        xmax = torch.clamp((pos2d[:, 0] + radius) // self.tile_length + 1, min=0, max=self.num_block_width).to(torch.int32)
        ymax = torch.clamp((pos2d[:, 1] + radius) // self.tile_length + 1, min=0, max=self.num_block_height).to(torch.int32)

        self.tile_count = torch.zeros((self.num_block_width, self.num_block_height),
                                      dtype=torch.int32, device=self.device)
        for pid in range(self.num_points):
            # px_min, py_min = xmin[pid].item(), ymin[pid].item()
            # px_max, py_max = xmax[pid].item(), ymax[pid].item()
            # if px_min == px_max or py_min == py_max:
            #     continue
            self.tile_count[xmin[pid]:xmax[pid], ymin[pid]:ymax[pid]] += 1
            torch.cuda.synchronize()

        indices_size = self.tile_count.sum()
        self.tile_count = torch.cumsum(
            torch.cat((torch.tensor([0], device=self.device), self.tile_count.flatten())),
            dim=0, dtype=torch.int32
        )[:-1].view((self.num_block_width, self.num_block_height))

        count = torch.zeros((self.num_block_width, self.num_block_height), dtype=torch.int32, device=self.device)
        self.tile_indices = torch.zeros(indices_size, dtype=torch.int32, device=self.device)
        for pid in range(self.num_points):
            px_min, py_min = xmin[pid].item(), ymin[pid].item()
            px_max, py_max = xmax[pid].item(), ymax[pid].item()
            if px_min == px_max or py_min == py_max:
                continue
            tiles = self.tile_count[px_min:px_max, py_min:py_max] + count[px_min:px_max, py_min:py_max]
            self.tile_indices[tiles] = pid
            torch.cuda.synchronize()
            count[px_min:px_max, py_min:py_max] += 1
            torch.cuda.synchronize()

        self.tile_count = self.tile_count.flatten()
        # print(f"end create tiles for {self.num_points} points")

    def depth_sort(self, depth: torch.tensor):
        for tid in range(self.num_block_width * self.num_block_height):
            prev, curr = self.get_indices_range(tid)
            sorted_indices = torch.argsort(depth[self.tile_indices[prev:curr]], descending=True)
            self.tile_indices[prev:curr] = self.tile_indices[prev:curr][sorted_indices]

    def convert_2dto1d(self, xi: int, yi: int):
        return yi + self.num_block_height * xi

    def convert_1dto2d(self, ti: int):
        xi = ti // self.num_block_height
        yi = ti - xi * self.num_block_height
        # print(f"ti({ti}) = xi({xi}), yi({yi})")
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

    def get_indices_range(self, tile_index: int) -> (int, int):
        assert (0 <= tile_index < self.num_block)
        next_tid = self.num_points if tile_index == self.num_block - 1 else self.tile_count[tile_index + 1]
        return self.tile_count[tile_index].item(), next_tid
