import pycolmap
import torch
import cv2
import numpy as np
from pathlib import Path
import random

from Utils.Camera import Camera
from Utils.PointCloud import PointCloud


class DataLoader:
    def __init__(self, image_dir: str, sfm_dir: str, seed: int = None):
        if seed is not None:
            random.seed(seed)

        self.image_dir = image_dir
        self.sfm_dir = sfm_dir
        self.database_file = f"{self.sfm_dir}/database.db"
        self.sparse_dir = f"{self.sfm_dir}/sparse"
        self.sfm_reconstruction = None
        self.train_image_ids = None
        self.validate_image_ids = None

    def extract_keypoint(self):
        sfm_reconstruction = pycolmap.Reconstruction()
        if Path(self.database_file).exists():
            sfm_reconstruction.read(self.sfm_dir)
            sfm_reconstruction.check()
            self.sfm_reconstruction = sfm_reconstruction
            return

        database_dir = Path(self.sfm_dir)
        database_dir.mkdir(parents=True, exist_ok=True)
        try:
            pycolmap.extract_features(self.database_file, self.image_dir, camera_model="SIMPLE_PINHOLE")
            pycolmap.match_exhaustive(self.database_file)

            sfm_reconstruction = pycolmap.incremental_mapping(self.database_file, self.image_dir, self.sparse_dir)
            sfm_reconstruction[0].write(self.sfm_dir)
            self.sfm_reconstruction = sfm_reconstruction
        except Exception as e:
            database_dir.unlink()
            print(f"Extract keypoint failed: {e}")

    def get_pcd(self) -> PointCloud:
        pcd = PointCloud()
        point_count = self.sfm_reconstruction.num_points3D()
        points = np.zeros((point_count, 3), dtype=np.float32)
        colors = np.zeros((point_count, 3), dtype=np.float32)
        for index, point_id in enumerate(self.sfm_reconstruction.point3D_ids()):
            point = self.sfm_reconstruction.points3D[point_id]
            points[index] = point.xyz
            colors[index] = point.color
        pcd.set_coords(points)
        pcd.set_colors(colors)
        return pcd

    def split_train_validate_data(self, ratio: float = 0.9):
        image_ids = self.sfm_reconstruction.reg_image_ids()
        num_train_ids = int(len(image_ids) * ratio)
        self.train_image_ids = image_ids[:num_train_ids]
        self.validate_image_ids = image_ids[num_train_ids:]

    def sample(self, is_train: bool = True, image_scale: float = 1) -> (Camera, torch.Tensor):
        if self.sfm_reconstruction is None:
            raise ValueError(f"SfM Reconstruction is None.")
        image_ids = self.train_image_ids if is_train else self.validate_image_ids
        sfm_image = self.sfm_reconstruction.images[random.choice(image_ids)]
        sfm_camera = self.sfm_reconstruction.cameras[sfm_image.camera_id]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_path = self.image_dir + "/" + sfm_image.name
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float().to(device)

        if image_scale != 1:
            image = torch.nn.functional.interpolate(image, scale_factor=image_scale, mode='bilinear')
        return Camera.from_sfm(sfm_image, sfm_camera, image.shape[0], image.shape[1]), image

