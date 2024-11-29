import torch


def l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - gt).mean()


# def ssim_loss(pred: torch.Tensor, gt: torch.tensor,
#          c1: torch.float32 = 0.0001, c2: torch.float32 = 0.0009,
#          window_size: torch.int16 = 11) -> torch.Tensor:
#     """
#     :param pred: prediction images in shape (W, H, C)
#     :param gt: ground truth images in shape (W, H, C)
#     :param c1: SSIM constant
#     :param c2: SSIM constant
#     :param window_size: size of window to do convolution
#     :return: final SSIM score between prediction and ground truth, have size (B, 1)
#     """
#     in_channel = pred.size(-1)
#     half_window_size = window_size // 2
#
#     # Create kernel
#     def gaussian_window(sigma=1.5):
#         # get 1D gauss filter of shape (window_size, 1)
#         coords = torch.arange(window_size).float() - half_window_size
#         g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
#         window_1d = g.unsqueeze(1) / g.sum()
#
#         # x * x.T to get 2D filter of shape (window_size, window_size)
#         window_2d = window_1d @ window_1d.transpose(1, 0)
#         window_2d /= window_2d.sum()
#         return window_2d.expand(in_channel, 1, window_size, window_size)
#
#     # Create 2D Gaussian filter from separable 1D filters
#     assert pred.shape == gt.shape
#     window = gaussian_window().to(pred.device)
#
#     # Mean and variance computation using Gaussian filter
#     pred = pred.permute((2, 1, 0))
#     gt = gt.permute((2, 1, 0))
#     mu1 = torch.conv2d(pred, window, padding=window_size // 2, groups=in_channel)
#     mu2 = torch.conv2d(gt, window, padding=window_size // 2, groups=in_channel)
#
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#
#     sigma1_sq = torch.conv2d(pred * pred, window, padding=half_window_size, groups=in_channel) - mu1_sq
#     sigma2_sq = torch.conv2d(gt * gt, window, padding=half_window_size, groups=in_channel) - mu2_sq
#     sigma12 = torch.conv2d(pred * gt, window, padding=half_window_size, groups=in_channel) - mu1_mu2
#
#     ans = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
#     return ans.mean()

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim_loss(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)