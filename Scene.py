import pycolmap
import torch
import numpy as np
from Utils.Camera import Camera
from GaussianModel import GaussianSphere

"""
pycolmap stores transform of image data in Image object and focal point data in Camera object
Hence, to get all information about how an image is taken by a camera, we need to access both of these classes
"""


class Scene:
    def __init__(self, sfm_scene: pycolmap.Reconstruction, args: dict):
        sfm_scene.check()

        # A camera can refer to multiple images, but I want to build bijective mapping here
        self.cameras = np.empty(sfm_scene.num_images(), dtype=object)

        param_count = GaussianSphere.get_parameter_number()
        self.point_capacity = int(args.get('capacity', 1e6))
        self.points = torch.empty(self.point_capacity, param_count, dtype=torch.float32)
        self.point_size = 0

        # load scene
        self.__load_scene(sfm_scene)

    def __load_scene(self, sfm_scene: pycolmap.Reconstruction):
        # load cameras
        idx = 0
        for _, image in sfm_scene.images.items():
            camera = sfm_scene.cameras[image.camera_id]
            self.cameras[idx] = Camera(camera, image)

        # load points
        num_points = sfm_scene.num_points3D()
        if num_points >= self.point_capacity:
            raise RuntimeError(f"[Scene] allocated {self.point_capacity} space but need {num_points} space")

        idx = 0
        for _, point in sfm_scene.points3D.items():
            self.points[idx] = GaussianSphere.parse_sfm_point(point)
            idx += 1
        self.point_size += num_points
