import torch


def l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - gt).mean()


def ssim_loss(pred: torch.Tensor, gt: torch.tensor,
         c1: torch.float32 = 0.0001, c2: torch.float32 = 0.0009,
         window_size: torch.int16 = 11) -> torch.Tensor:
    """
    :param pred: prediction images in shape (W, H, C)
    :param gt: ground truth images in shape (W, H, C)
    :param c1: SSIM constant
    :param c2: SSIM constant
    :param window_size: size of window to do convolution
    :return: final SSIM score between prediction and ground truth, have size (B, 1)
    """
    in_channel = pred.size(-1)
    half_window_size = window_size // 2

    # Create kernel
    def gaussian_window(sigma=1.5):
        # get 1D gauss filter of shape (window_size, 1)
        coords = torch.arange(window_size).float() - half_window_size
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        window_1d = g.unsqueeze(1) / g.sum()

        # x * x.T to get 2D filter of shape (window_size, window_size)
        window_2d = window_1d @ window_1d.transpose(1, 0)
        window_2d /= window_2d.sum()
        return window_2d.expand(in_channel, 1, window_size, window_size)

    # Create 2D Gaussian filter from separable 1D filters
    print(pred.shape, gt.shape)
    assert pred.shape == gt.shape
    window = gaussian_window().to(pred.device)

    # Mean and variance computation using Gaussian filter
    pred = pred.permute((2, 1, 0))
    gt = gt.permute((2, 1, 0))
    mu1 = torch.conv2d(pred, window, padding=window_size // 2, groups=in_channel)
    mu2 = torch.conv2d(gt, window, padding=window_size // 2, groups=in_channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.conv2d(pred * pred, window, padding=half_window_size, groups=in_channel) - mu1_sq
    sigma2_sq = torch.conv2d(gt * gt, window, padding=half_window_size, groups=in_channel) - mu2_sq
    sigma12 = torch.conv2d(pred * gt, window, padding=half_window_size, groups=in_channel) - mu1_mu2

    ans = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ans.mean()
