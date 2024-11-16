import torch
import numpy as np


def torch2numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr

    if not isinstance(arr, torch.Tensor):
        raise ValueError(f"Input is not type torch.Tensor.")
    if arr.is_cuda:
        return arr.cpu().numpy()
    elif arr.device == "cpu":
        return arr.numpy()
    else:
        raise NotImplementedError("Target device check is not implemented")


def numpy2torch(arr, device=None):
    if isinstance(arr, torch.Tensor):
        return arr

    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Input is not type np.ndarray.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.from_numpy(arr, device=device)

