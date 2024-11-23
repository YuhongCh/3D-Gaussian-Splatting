import torch


def gaussian_distribution(center: torch.tensor, covariance: torch.tensor, points: torch.tensor, batch_size: int = -1):
    """
    Compute the probability at point with Gaussian Distribution of given center and covariance
    :param center: center of gaussian distribution (shape Nx2)
    :param covariance: covariance of gaussian distribution (shape Nx2x2)
    :param points: points in world position (shape M1xM2x2)
    :param batch_size: for optimization use, batch the number of points to compute at once, if -1, assume no batch
    :return: probability of points locate in distribution with shape NxM
    """
    if batch_size == -1:
        batch_size = points.shape[0]

    # inverse covariance, resulting shape Nx2x2
    inv_covariance = torch.linalg.inv(covariance)

    # Prepare for batching
    N, M1, M2 = center.shape[0], points.shape[0], points.shape[1]
    result = torch.zeros((N, M1 * M2), device=points.device, dtype=torch.float32)

    # Flatten points for easier batching, resulting shape (M1 * M2, 2)
    points_flat = points.reshape(-1, 2)

    for si in range(0, points_flat.shape[0], batch_size):
        batch_points = points_flat[si:si + batch_size, :]                # Shape Bx2
        local_points = center[:, None, :] - batch_points[None, :, :]  # Shape NxBx2
        expon = torch.sum((local_points @ inv_covariance) * local_points, dim=-1) # Shape NxB
        result[:, si:si+batch_size] = torch.exp(-0.5 * expon)
    return result.view(N, M1, M2)
