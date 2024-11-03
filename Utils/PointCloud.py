import numpy as np
import torch


class PointCloud:
    def __init__(self, size: int = 1000, device: str = None):
        """
        :param size: number of point in pcd, each has coord XYZ and color RGBA
        """
        self._device = device
        self._size = size
        self._coords = np.zeros((self._size, 3), dtype=np.float32)
        self._colors = np.zeros((self._size, 4), dtype=np.int8)   # [0, 255]

    def __repr__(self):
        return f'coords: {self._coords.shape[0]} \t colors: {self._colors.shape[0]}'

    @property
    def coords(self):
        return self._coords

    @property
    def colors(self):
        return self._colors

    @property
    def size(self):
        return self._size

    @property
    def device(self):
        return self._device

    def resize(self, size: int):
        """
        If current size is smaller than given size, expand the array
        else shrink the array from behind
        :param size: new array size
        """
        if self._size == size:
            return
        elif self._size < size:
            extra_size = size - self._size
            self._coords = np.concatenate((self._coords, np.zeros((extra_size, 3), dtype=np.float32)), axis=0)
            self._colors = np.concatenate((self._colors, np.zeros((extra_size, 4), dtype=np.int8)), axis=0)
        else:
            self._coords = self._coords[:size, :]
            self._colors = self._colors[:size, :]
        self._size = size

    def set_coords(self, coords: np.ndarray):
        """
        Set coordinate of points and dynamically allocate space if needed
        NOTE: size of color array will also change in size accordingly
        :param coords: np array of shape (N, 3)
        """
        assert(len(coords.shape) == 2 and coords.shape[1] == 3)
        self.resize(coords.shape[0])
        self._coords = coords

    def set_colors(self, colors: np.ndarray):
        """
        Set color of points and dynamically allocate space if needed
        NOTE: size of coords array will also change in size accordingly
        :param colors: np array of shape (N, 4)
        """
        assert (len(colors.shape) == 2 and colors.shape[1] == 4)
        self.resize(colors.shape[0])
        self._colors = colors

    def random_sample(self, num_points: int):
        """
        Sample a random subset of this PointCloud.
        :param num_points: maximum number of points to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud with size of num_points
        """
        if self._size <= num_points:
            return self
        pcd = PointCloud(num_points)
        indices = np.random.choice(self._size, size=(num_points,), replace=False)
        pcd.set_coords(self.coords[indices])
        pcd.set_colors(self.colors[indices])

    @staticmethod
    def combine(pcd1: "PointCloud", pcd2: "PointCloud"):
        """
        combine two PointCloud object to get a new PointCloud object with data from both of them
        :param pcd1: a PointCloud object
        :param pcd2: a PointCloud object
        :return: new PointCloud object with data from both pcd1 and pcd2
        """
        size = pcd1.size + pcd2.size
        pcd = PointCloud(size)
        pcd.set_coords(np.concatenate((pcd1.coords, pcd2.coords), axis=0))
        pcd.set_colors(np.concatenate((pcd1.colors, pcd2.colors), axis=0))
        return pcd


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Below referenced from https://github.com/openai/point-e/blob/main/point_e/util/ply_util.py 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import io
import struct
from contextlib import contextmanager
from typing import BinaryIO, Iterator, Optional


def write_ply(raw_f: BinaryIO, coords: np.ndarray, rgb: Optional[np.ndarray] = None, faces: Optional[np.ndarray] = None):
    """
    Write a PLY file for a mesh or a point cloud.

    :param coords: an [N x 3] array of floating point coordinates.
    :param rgb: an [N x 3] array of vertex colors, in the range [0.0, 1.0].
    :param faces: an [N x 3] array of triangles encoded as integer indices.
    """
    with buffered_writer(raw_f) as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(bytes(f"element vertex {len(coords)}\n", "ascii"))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        if rgb is not None:
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")
        if faces is not None:
            f.write(bytes(f"element face {len(faces)}\n", "ascii"))
            f.write(b"property list uchar int vertex_index\n")
        f.write(b"end_header\n")

        if rgb is not None:
            rgb = (rgb * 255.499).round().astype(int)
            vertices = [
                (*coord, *rgb)
                for coord, rgb in zip(
                    coords.tolist(),
                    rgb.tolist(),
                )
            ]
            format = struct.Struct("<3f3B")
            for item in vertices:
                f.write(format.pack(*item))
        else:
            format = struct.Struct("<3f")
            for vertex in coords.tolist():
                f.write(format.pack(*vertex))

        if faces is not None:
            format = struct.Struct("<B3I")
            for tri in faces.tolist():
                f.write(format.pack(len(tri), *tri))


@contextmanager
def buffered_writer(raw_f: BinaryIO) -> Iterator[io.BufferedIOBase]:
    if isinstance(raw_f, io.BufferedIOBase):
        yield raw_f
    else:
        f = io.BufferedWriter(raw_f)
        yield f
        f.flush()
