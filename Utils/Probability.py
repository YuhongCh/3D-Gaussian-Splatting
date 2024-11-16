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
    inv_covariance = torch.linalg.inv(covariance)

    # want to change the center (Nx2) and points (M1xM2x2) to (NxM1xM2x2)
    # change the center to shape Nx1x1x2 and points to 1xM1xM2x2 to get NxM1xM2x2
    # resulting shape NxM1xM2x2
    local_points = center.unsqueeze(1).unsqueeze(1) - points.unsqueeze(0)

    # x^TSx => (NxM1xM2x2)(Nx2x2)(NxM1xM2x2) => Nx((M1xM2x2)(2x2)(M1xM2x2)) =>
    # Nx((M1xM2x1x2)(1x1x2x2)(M1xM2x2x1)) => Nx(M1xM2x1x1) => NxM1xM2
    local_points = local_points.unsqueeze(4)
    inv_covariance = inv_covariance.unsqueeze(1).unsqueeze(2)
    # print(center.shape, points.shape, local_points.shape, inv_covariance.shape)
    exponential = local_points.permute((0, 1, 2, 4, 3)) @ inv_covariance @ local_points
    N, M1, M2, _, _ = exponential.shape
    exponential = exponential.reshape((N, M1, M2))
    ans = torch.exp(-0.5 * exponential)


    # assert(target == ans[0, 0, 0])
    return ans

