import pycolmap
import torch
import numpy as np
import torch.nn as nn
import Utils.Transform as Transform

from math import tan, atan


class Camera(nn.Module):
    def __init__(self, position, rotation, fovX, fovY, target_image, znear=0.1, zfar=100):
        super(Camera, self).__init__()

        self.transform = Transform(position, rotation=rotation)
        self.fovX = fovX
        self.fovY = fovY
        self.znear = znear
        self.zfar = zfar

        self.target_image = target_image
        self.height = self.target_image.shape[0]
        self.width = self.target_image.shape[1]

        self.camera2world_matrix = self.transform.get_model2world_matrix()
        self.word2camera_matrix = self.transform.get_world2model_matrix()
        self.projection_matrix = self.get_projection_matrix(self.znear, self.zfar, self.fovX, self.fovY)

    @staticmethod
    def get_projection_matrix(znear: float, zfar: float, fovX: float, fovY: float):
        """
        :param znear: screen distance to render the image
        :param zfar: maximum distance to have object visible on screen
        :param fovX: field of view in X axis, represent width
        :param fovY: field of view in Y axis, represent height
        :return: projection matrix to project points in view frustum onto screen image
        """
        right = znear * tan(0.5 * fovX)
        top = znear * tan(0.5 * fovY)
        left = -right
        bottom = -top
        return np.array([
            [2 * znear / (right - left), 0, (left + right) / (right - left), 0],
            [0, 2 * znear / (top - bottom), (top + bottom) / (top - bottom), 0],
            [0, 0, zfar / (zfar - znear), -zfar * znear / (zfar - znear)],
            [0, 0, 1, 0]
        ], dtype=float)

    @staticmethod
    def focal2fov(focal, pixels):
        """
        :param focal: focal length of camera
        :param pixels: height or width of camera screen in terms of number of pixels
        :return: fov of camera
        """
        return 2 * atan(pixels / (2 * focal))

    @staticmethod
    def fov2focal(fov, pixels):
        """
        similar as focal2fov
        """
        return pixels / (2 * tan(fov / 2))
