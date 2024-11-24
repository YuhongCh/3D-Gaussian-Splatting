import torch


def gaussian_distribution(center: torch.tensor, covariance: torch.tensor, points: torch.tensor):
    """
    Compute the probability at point with Gaussian Distribution of given center and covariance
    :param center: center of gaussian distribution (shape Nx2)
    :param covariance: covariance of gaussian distribution (shape Nx2x2)
    :param points: points in world position (shape M1xM2x2)
    :return: probability of points locate in distribution with shape NxM
    """

    # inverse covariance, resulting shape Nx2x2
    N, M1, M2 = center.shape[0], points.shape[0], points.shape[1]
    inv_covariance = torch.linalg.inv(covariance)

    # Flatten points for easier batching, resulting shape (M1 * M2, 2)
    inv_covariance = torch.linalg.inv(covariance)
    points_flat = points.reshape(-1, 2)
    local_points = center[:, None, :] - points_flat[None, :, :]
    expon = torch.sum((local_points @ inv_covariance) * local_points, dim=-1)
    result = torch.exp(-0.5 * expon)
    return result.view(N, M1, M2)
