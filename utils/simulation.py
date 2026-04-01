import numpy as np
import odl
import torch
from odl.contrib.torch import OperatorFunction


class DegradationSimulatorCT:
    def __init__(self, device: torch.device | None = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_projection()
        self.i0_full = torch.tensor(1e6, device=self.device, dtype=torch.float32)
        self.sigma_e_gassian = torch.tensor(1.0, device=self.device, dtype=torch.float32)

    def _initialize_projection(self) -> None:
        image_height = 512
        image_width = 512
        reco_space = odl.uniform_discr(
            min_pt=[-image_height / 4, -image_width / 4],
            max_pt=[image_height / 4, image_width / 4],
            shape=[image_height, image_width],
            dtype="float32",
        )
        angle_partition = odl.uniform_partition(0, 2 * np.pi, 720)
        detector_partition = odl.uniform_partition(-360, 360, 1024)
        geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=1270, det_radius=870)
        self.forward_op = odl.tomo.RayTransform(reco_space, geometry)
        self.filtered_backward_op = odl.tomo.fbp_op(self.forward_op)

    @staticmethod
    def _hu_to_attenuation(ct_data: torch.Tensor) -> torch.Tensor:
        ct_data = ct_data.clone()
        ct_data[ct_data < -1000] = -1000
        return (ct_data / 1000.0 + 1.0) * 0.02

    @staticmethod
    def _attenuation_to_hu(ct_data: torch.Tensor) -> torch.Tensor:
        attenuation = (ct_data / 0.02 - 1.0) * 1000.0
        return torch.clamp(attenuation, -1000.0, 3072.0)

    @staticmethod
    def _hu_to_unit(ct_data: torch.Tensor) -> torch.Tensor:
        return (ct_data + 1024.0) / 4096.0

    @staticmethod
    def _unit_to_hu(ct_data: torch.Tensor) -> torch.Tensor:
        return ct_data * 4096.0 - 1024.0

    def _normalize(self, ct_data: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "hu2attenuation":
            return self._hu_to_attenuation(ct_data)
        if mode == "unit2attenuation":
            return self._hu_to_attenuation(self._unit_to_hu(ct_data))
        if mode == "attenuation2hu":
            return self._attenuation_to_hu(ct_data)
        if mode == "attenuation2unit":
            return self._hu_to_unit(self._attenuation_to_hu(ct_data))
        raise ValueError(f"Unsupported mode: {mode}")

    def degrade(self, ct_images: torch.Tensor, delta_t: torch.Tensor, ct_unit: str = "unit") -> torch.Tensor:
        with torch.no_grad():
            if not torch.is_tensor(delta_t):
                delta_t = torch.tensor(float(delta_t), device=self.device, dtype=torch.float32)
            attenuation = self._normalize(ct_images, f"{ct_unit}2attenuation")
            projection = OperatorFunction.apply(self.forward_op, attenuation)
            alpha = torch.exp(-5.0 * delta_t)
            expected_intensity = (alpha * self.i0_full) * torch.exp(-projection)
            photon_counts = torch.poisson(expected_intensity)
            sigma_total = self.sigma_e_gassian * torch.sqrt(400.0 * delta_t)
            electronics_noise = sigma_total * torch.randn_like(photon_counts, device=self.device)
            transmission = (photon_counts + electronics_noise) / (alpha * self.i0_full)
            transmission = torch.clamp(transmission, min=1e-6)
            projection_recon = -torch.log(transmission)
            attenuation_recon = OperatorFunction.apply(self.filtered_backward_op, projection_recon)
            return self._normalize(attenuation_recon, f"attenuation2{ct_unit}")


class DegradationSimulatorMR:
    def __init__(self, device: torch.device | None = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reduction_rate = torch.tensor(0.005, device=self.device, dtype=torch.float32)

    def degrade(self, mag_img: torch.Tensor, delta_t: float | torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if not torch.is_tensor(delta_t):
                delta_t = torch.tensor(float(delta_t), device=mag_img.device, dtype=torch.float32)
            std = torch.sqrt(delta_t * self.reduction_rate)
            n1 = torch.normal(0.0, float(std), size=mag_img.shape, device=mag_img.device, dtype=mag_img.dtype)
            n2 = torch.normal(0.0, float(std), size=mag_img.shape, device=mag_img.device, dtype=mag_img.dtype)
            return torch.sqrt((mag_img + n1) ** 2 + n2 ** 2)
