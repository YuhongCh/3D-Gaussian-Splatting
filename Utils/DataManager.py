import pycolmap
import torch
import numpy as np
import open3d as o3d
from pathlib import Path

from Utils.Camera import Camera
from Utils.PointCloud import PointCloud


class DataManager:
    def __init__(self, sfm_dir: str, image_dir: str = None, ply_file: str = None):
        self.image_dir = image_dir
        self.sfm_dir = sfm_dir
        self.ply_file = ply_file

        self.pcd_data = PointCloud()
        self.image_data = torch.empty()
        self.cam_data = {}

        is_success = self._try_load_sfm(sfm_dir) or self._try_load_image(image_dir)
        is_success = is_success and self._try_load_ply(self.ply_file)
        if not is_success:
            raise OSError("Failed to load the data")

    def _try_load_ply(self, ply_file: str) -> bool:
        if ply_file is None:
            print("ply_file is None")
            return False
        if not Path(ply_file).exists():
            print(f"[Warning] Cannot find the ply_file entered: f{ply_file}")
            return False

        pcd = o3d.io.read_point_cloud(ply_file)
        self.pcd_data.set_coords(np.asarray(pcd.points))
        self.pcd_data.set_colors(np.asarray(pcd.colors))
        print("Success load ply data")
        return True

    def _try_load_image(self, image_dir: str) -> bool:
        if image_dir is None:
            print("image_dir is None")
            return False

        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            self.image_dir = None
            print(f"[Warning] Cannot find the image_dir entered: f{image_dir}")
            return False

        sfm_dir_str = "sfm_database"
        print(f"Success find the image data, start parsing and save to f{sfm_dir_str}")

        sfm_dir = Path(sfm_dir_str)
        sfm_dir.mkdir()
        sfm_database_path = sfm_dir / "database.db"
        pycolmap.extract_features(sfm_database_path, self.image_dir)
        pycolmap.match_exhaustive(sfm_database_path)
        maps = pycolmap.incremental_mapping(sfm_database_path, self.image_dir, sfm_dir)
        maps[0].write(sfm_dir)
        return self._try_load_sfm(sfm_dir_str)

    def _try_load_sfm(self, sfm_dir: str) -> bool:
        if sfm_dir is None:
            print("sfm_dir is None")
            return False

        self.sfm_dir = Path(sfm_dir)
        if not self.sfm_dir.exists():
            self.sfm_dir = None
            print(f"[Warning] Cannot find the sfm_dir entered: f{sfm_dir}")
            return False
        sfm_scene = pycolmap.Reconstruction()
        sfm_scene.read(sfm_dir)
        sfm_scene.check()
        print(f"Success load the sfm data, start parsing...")
        return self._try_parse_sfm(sfm_scene)

    def _try_parse_sfm(self, sfm_scene: pycolmap.Reconstruction) -> bool:
        # parse point cloud data
        points = np.zeros((sfm_scene.num_points3D(), 3), dtype=np.float32)
        colors = np.zeros((sfm_scene.num_points3D(), 3), dtype=np.float32)
        for index, point_id in enumerate(sfm_scene.point3D_ids()):
            point = sfm_scene.points3D(point_id)
            points[index] = point.xyz
            colors[index] = point.color
        self.pcd_data.set_coords(points)
        self.pcd_data.set_colors(colors)

        for cid, camera in sfm_scene.cameras.items():
          self.cam_data[cid] = Camera()

