import torch


def gaussian_distribution(center: torch.tensor, covariance: torch.tensor, points: torch.tensor):
    """
    Compute the probability at point with Gaussian Distribution of given center and covariance
    :param center: center of gaussian distribution (shape Nx2)
    :param covariance: covariance of gaussian distribution (shape Nx2x2)
    :param points: points in world position (shape Mx2)
    :return: probability of points locate in distribution with shape NxM
    """
    # inverse covariance, resulting shape Nx2x2
    inv_covariance = torch.linalg.inv(covariance)

    # change the points to shape 1xMx2, center to shape Nx1x2 to subtract
    # resulting shape MxNx2
    local_points = points.unsqueeze(0) - center.unsqueeze(1)

    # x^TSx => (MxNx2)(Nx2x2)(MxNx2) => (NxMx2)(Nx2x2)(NxMx2)
    # (NxMx2)(Nx2x2)(NxMx2) => Nx((Mx2)(2x2)(Mx2) => (Mx1x2)(2x2)(Mx2x1) => (Mx1x1)) => NxM
    local_points = local_points.permute((0, 1))
    exponential = local_points.unsqueeze(1) @ inv_covariance @ local_points.unsqueeze(2)
    N = center.shape[0], M = points.shape[0]
    assert (exponential.shape == (N, M, 1, 1) and f"exponential has shape {exponential.shape}")
    exponential.reshape((N, M))
    return torch.exp(-0.5 * exponential)

