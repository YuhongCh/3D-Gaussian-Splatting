import numpy as np
import torch

class Transform:
    def __init__(self, position, rotation=None, scale=None):
        """
        :param position: position of given object in (x,y,z)
        :param rotation: rotation of given object in quaternion (w,x,y,z), default (1,0,0,0)
        :param scale: scale of given object in (x,y,z)
        """
        self.position = position

        if rotation is None:
            rotation = np.array([1, 0, 0, 0])
        elif rotation.shape[0] != 4:
            raise ValueError(f"Rotation should have shape 4 but get {rotation.shape[0]}")
        self.rotation = rotation

        if scale is None:
            scale = np.array([0, 0, 1])
        elif scale.shape[0] != 3:
            raise ValueError(f"Scale should have shape 3 but get {scale.shape[0]}")
        self.scale = scale

    @staticmethod
    def get_rotation_matrix(w: float, x: float, y: float, z: float):
        return np.array([
            [2 * (w * w + x * x) - 1, 2 * (x * y - w * z),     2 * (x * z + w * y),     0],
            [2 * (x * y + w * z),     2 * (w * w + y * y) - 1, 2 * (y * z - w * x),     0],
            [2 * (x * z - w * y),     2 * (y * z + w * x),     2 * (w * w + z * z) - 1, 0],
            [0,                       0,                       0,                       1]
        ], dtype=np.float32)

    @staticmethod
    def get_rotation_matrices(q_array, is_cuda=False):
        """
        convert quaternions q_array to rotation matrices.
        :param q_array: array of quaternion with shape (N, 4), MUST accept slicing here
        :param is_cuda: if q_array is in torch, this allows use of cuda
        :return: matrices of shape (N, 4, 4)
        """
        N = q_array.shape[0]
        device = None
        if is_cuda:
            device = "cuda"
        rot = torch.zeros((N, 4, 4), dtype=torch.float32, device=device)
        rot[:, 0, 0] = 2 * (q_array[:, 0] * q_array[:, 0] + q_array[:, 1] * q_array[:, 1]) - 1
        rot[:, 0, 1] = 2 * (q_array[:, 1] * q_array[:, 2] - q_array[:, 0] * q_array[:, 3])
        rot[:, 0, 2] = 2 * (q_array[:, 1] * q_array[:, 3] + q_array[:, 0] * q_array[:, 2])
        rot[:, 1, 0] = 2 * (q_array[:, 1] * q_array[:, 2] + q_array[:, 0] * q_array[:, 3])
        rot[:, 1, 1] = 2 * (q_array[:, 0] * q_array[:, 0] + q_array[:, 1] * q_array[:, 1]) - 1
        rot[:, 1, 2] = 2 * (q_array[:, 2] * q_array[:, 3] - q_array[:, 0] * q_array[:, 1])
        rot[:, 2, 0] = 2 * (q_array[:, 1] * q_array[:, 3] - q_array[:, 0] * q_array[:, 2])
        rot[:, 2, 1] = 2 * (q_array[:, 2] * q_array[:, 3] + q_array[:, 0] * q_array[:, 1])
        rot[:, 2, 2] = 2 * (q_array[:, 0] * q_array[:, 0] + q_array[:, 1] * q_array[:, 1]) - 1
        rot[:, 3, 3] = 1
        return rot

    @staticmethod
    def get_inv_rotation_matrix(w: float, x: float, y: float, z: float):
        return Transform.get_rotation_matrix(w, -x, -y, -z)

    @staticmethod
    def get_scale_matrix(x: float, y: float, z: float):
        return np.array([
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [0, 0, z, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    @staticmethod
    def get_scale_matrices(s_array, is_cuda=False):
        N = s_array.shape[0]
        device = None
        if is_cuda:
            device = "cuda"
        scale = torch.zeros((N, 4, 4), dtype=torch.float32, device=device)
        scale[:, 0, 0] = s_array[:, 0]
        scale[:, 1, 1] = s_array[:, 1]
        scale[:, 2, 2] = s_array[:, 2]
        scale[:, 3, 3] = 1
        return scale

    @staticmethod
    def get_inv_scale_matrix(x: float, y: float, z: float):
        return Transform.get_scale_matrix(1.0 / x, 1.0 / y, 1.0 / z)

    @staticmethod
    def get_translation_matrix(x: float, y: float, z: float):
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ], dtype=np.float32)

    @staticmethod
    def get_inv_translation_matrix(x: float, y: float, z: float):
        return Transform.get_inv_translation_matrix(-x, -y, -z)

    def get_model2world_matrix(self):
        return (self.get_translation_matrix(self.position[0], self.position[1], self.position[2]) @
                self.get_rotation_matrix(self.rotation[0], self.rotation[1], self.rotation[2], self.rotation[3]) @
                self.get_scale_matrix(self.scale[0], self.scale[1], self.scale[2]))

    def get_world2model_matrix(self):
        return (self.get_inv_translation_matrix(self.position[0], self.position[1], self.position[2]) @
                self.get_inv_rotation_matrix(self.rotation[0], self.rotation[1], self.rotation[2], self.rotation[3]) @
                self.get_inv_scale_matrix(self.scale[0], self.scale[1], self.scale[2]))

    def get_covariance(self):
        rotation = self.get_rotation_matrix(self.rotation[0], self.rotation[1], self.rotation[2], self.rotation[3])
        scale = self.get_scale_matrix(self.scale[0], self.scale[1], self.scale[2])
        return rotation @ scale @ (rotation @ scale).transpose()
