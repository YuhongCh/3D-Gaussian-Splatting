from Utils.ContainerUtils import *


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
            scale = np.array([1, 1, 1])
        elif scale.shape[0] != 3:
            raise ValueError(f"Scale should have shape 3 but get {scale.shape[0]}")
        self.scale = scale

    @staticmethod
    def get_rotation_matrix(w: float, x: float, y: float, z: float, use_torch: bool = False):
        rot = np.array([
            [2 * (w * w + x * x) - 1, 2 * (x * y - w * z),     2 * (x * z + w * y),     0],
            [2 * (x * y + w * z),     2 * (w * w + y * y) - 1, 2 * (y * z - w * x),     0],
            [2 * (x * z - w * y),     2 * (y * z + w * x),     2 * (w * w + z * z) - 1, 0],
            [0,                       0,                       0,                       1]
        ], dtype=np.float32)
        if use_torch:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            rot = torch.from_numpy(rot).to(device)
        return rot

    @staticmethod
    def get_rotation_matrices(q_array, use_torch: bool = False):
        """
        convert quaternions q_array to rotation matrices. q_array elements MUST be in form [w, x, y, z]
        :param q_array: array of quaternion with shape (N, 4), MUST accept slicing here
        :param use_torch: decide whether return the data in torch or numpy
        :return: matrices of shape (N, 4, 4)
        """
        N = q_array.shape[0]
        if use_torch:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            rot = torch.zeros((N, 4, 4), dtype=torch.float32, device=device)
        else:
            rot = np.zeros((N, 4, 4), dtype=np.float32)

        rot[:, 0, 0] = 2 * (q_array[:, 0] * q_array[:, 0] + q_array[:, 1] * q_array[:, 1]) - 1
        rot[:, 0, 1] = 2 * (q_array[:, 1] * q_array[:, 2] - q_array[:, 0] * q_array[:, 3])
        rot[:, 0, 2] = 2 * (q_array[:, 1] * q_array[:, 3] + q_array[:, 0] * q_array[:, 2])
        rot[:, 1, 0] = 2 * (q_array[:, 1] * q_array[:, 2] + q_array[:, 0] * q_array[:, 3])
        rot[:, 1, 1] = 2 * (q_array[:, 0] * q_array[:, 0] + q_array[:, 2] * q_array[:, 2]) - 1
        rot[:, 1, 2] = 2 * (q_array[:, 2] * q_array[:, 3] - q_array[:, 0] * q_array[:, 1])
        rot[:, 2, 0] = 2 * (q_array[:, 1] * q_array[:, 3] - q_array[:, 0] * q_array[:, 2])
        rot[:, 2, 1] = 2 * (q_array[:, 2] * q_array[:, 3] + q_array[:, 0] * q_array[:, 1])
        rot[:, 2, 2] = 2 * (q_array[:, 0] * q_array[:, 0] + q_array[:, 3] * q_array[:, 3]) - 1
        rot[:, 3, 3] = 1

        return rot

    @staticmethod
    def get_inv_rotation_matrix(w: float, x: float, y: float, z: float, use_torch: bool = False):
        return Transform.get_rotation_matrix(w, -x, -y, -z, use_torch=use_torch)

    @staticmethod
    def get_scale_matrix(x: float, y: float, z: float, use_torch: bool = False):
        scale = np.array([
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [0, 0, z, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        if use_torch:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            scale = torch.from_numpy(scale).to(device)
        return scale

    @staticmethod
    def get_scale_matrices(s_array, use_torch: bool = False):
        N = s_array.shape[0]
        if use_torch:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            scale = torch.zeros((N, 4, 4), dtype=torch.float32, device=device)
        else:
            scale = np.zeros((N, 4, 4), dtype=torch.float32)

        scale[:, 0, 0] = s_array[:, 0]
        scale[:, 1, 1] = s_array[:, 1]
        scale[:, 2, 2] = s_array[:, 2]
        scale[:, 3, 3] = 1

        return scale

    @staticmethod
    def get_inv_scale_matrix(x: float, y: float, z: float, use_torch: bool = False):
        if x == 0 or y == 0 or z == 0:
            raise RuntimeError(f"Cannot divide by zero with components ({x},{y},{z})")
        return Transform.get_scale_matrix(1.0 / x, 1.0 / x, 1.0 / z, use_torch=use_torch)

    @staticmethod
    def get_translation_matrix(x: float, y: float, z: float, use_torch: bool = False):
        translate = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        if use_torch:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            translate = torch.from_numpy(translate).to(device)
        return translate

    @staticmethod
    def get_inv_translation_matrix(x: float, y: float, z: float, use_torch: bool = False):
        return Transform.get_translation_matrix(-x, -y, -z, use_torch=use_torch)

    def get_model2world_matrix(self, use_torch: bool = False):
        m2w = (self.get_translation_matrix(self.position[0], self.position[1], self.position[2]) @
               self.get_rotation_matrix(self.rotation[0], self.rotation[1], self.rotation[2], self.rotation[3]) @
               self.get_scale_matrix(self.scale[0], self.scale[1], self.scale[2]))
        if use_torch:
            return numpy2torch(m2w)
        return m2w

    def get_world2model_matrix(self, use_torch: bool = False):
        w2m = (self.get_inv_translation_matrix(self.position[0], self.position[1], self.position[2]) @
               self.get_inv_rotation_matrix(self.rotation[0], self.rotation[1], self.rotation[2], self.rotation[3]) @
               self.get_inv_scale_matrix(self.scale[0], self.scale[1], self.scale[2]))
        if use_torch:
            return numpy2torch(w2m)
        return w2m

    def get_covariance(self, use_torch: bool = False):
        rotation = self.get_rotation_matrix(self.rotation[0], self.rotation[1], self.rotation[2], self.rotation[3])
        scale = self.get_scale_matrix(self.scale[0], self.scale[1], self.scale[2])
        rs = rotation @ scale
        cov = rs @ rs.T

        if use_torch:
            return numpy2torch(cov)
        return cov

    @staticmethod
    def get_world2view2(R, t, translate=np.array([.0,.0,.0]), scale=1.0):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)



