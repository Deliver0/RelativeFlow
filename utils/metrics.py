from math import exp

import lpips
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class MetricComputer:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.loss_fn_alex = lpips.LPIPS(net="alex").to(device)

    @staticmethod
    def _auto_correct_background(image: torch.Tensor, percentile: float = 5.0) -> torch.Tensor:
        bg_value = torch.quantile(image.flatten(), percentile / 100.0)
        return torch.clamp(image - bg_value, 0.0, 1.0)

    def _preprocess(self, image: torch.Tensor, mode: str) -> torch.Tensor:
        image = image.clone()
        if mode == "MR":
            max_val = image.max()
            if max_val > 0:
                image = image / max_val
            image = self._auto_correct_background(image)
        return image

    @staticmethod
    def _mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return ((img1 - img2) ** 2).mean()

    def compute_rmse(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        return torch.sqrt(self._mse(img1, img2)).item()

    def compute_psnr(self, img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
        mse = self._mse(img1, img2)
        return 10.0 * torch.log10(torch.tensor(data_range**2, device=img1.device) / mse).item()

    @staticmethod
    def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
        values = [exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
        gauss = torch.tensor(values)
        return gauss / gauss.sum()

    @classmethod
    def _create_window(cls, window_size: int, channel: int) -> torch.Tensor:
        window_1d = cls._gaussian(window_size, 1.5).unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        return Variable(window_2d.expand(channel, 1, window_size, window_size).contiguous())

    def compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0, window_size: int = 11) -> float:
        if img1.dim() == 2:
            side = img1.shape[-1]
            img1 = img1.view(1, 1, side, side)
            img2 = img2.view(1, 1, side, side)
        window = self._create_window(window_size, img1.shape[1]).type_as(img1)
        mu1 = F.conv2d(img1, window, padding=window_size // 2)
        mu2 = F.conv2d(img2, window, padding=window_size // 2)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2) - mu1_mu2
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim_map.mean().item()

    def compute_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        img1 = img1 * 2.0 - 1.0
        img2 = img2 * 2.0 - 1.0
        img1_rgb = img1.repeat(1, 3, 1, 1)
        img2_rgb = img2.repeat(1, 3, 1, 1)
        return self.loss_fn_alex(img1_rgb, img2_rgb).item()

    def compute_all(self, prediction: torch.Tensor, target: torch.Tensor, mode: str) -> tuple[float, float, float, float]:
        prediction = self._preprocess(prediction, mode)
        target = self._preprocess(target, mode)
        return (
            self.compute_psnr(prediction, target),
            self.compute_ssim(prediction, target),
            self.compute_rmse(prediction, target),
            self.compute_lpips(prediction, target),
        )
