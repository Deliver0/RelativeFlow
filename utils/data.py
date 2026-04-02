from pathlib import Path
from typing import List, Sequence, Tuple

import nibabel as nib
import numpy as np
import pydicom
import torch
from pydicom.dataset import FileDataset
from torch.utils.data import Dataset


def read_paired_paths(file_path: str | Path) -> List[Tuple[str, str]]:
    with open(file_path, "r", encoding="utf-8") as handle:
        pairs = [tuple(line.strip().split(",", 1)) for line in handle if line.strip()]
    return [(str(a), str(b)) for a, b in pairs]


def read_path_list(file_path: str | Path) -> List[str]:
    paths: List[str] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if "," in stripped:
                _, target_path = stripped.split(",", 1)
                paths.append(str(target_path))
            else:
                paths.append(str(stripped))
    return paths


def _downsample_2d(image: np.ndarray, image_size: int) -> np.ndarray:
    h, w = image.shape
    step_h = max(1, h // image_size)
    step_w = max(1, w // image_size)
    return image[::step_h, ::step_w]


def load_ct_dicom(path: str | Path, image_size: int) -> np.ndarray:
    dcm = pydicom.dcmread(str(path), force=True)
    image = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    image = image * slope + intercept
    image = np.clip(image, -1024.0, 3072.0)
    image = (image + 1024.0) / 4096.0
    image = _downsample_2d(image, image_size)
    return np.expand_dims(image.astype(np.float32), axis=0)


def save_ct_dicom(tensor: torch.Tensor, reference_path: str | Path, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.squeeze().detach().cpu().numpy().astype(np.float32)
    hu_values = np.clip(image * 4096.0 - 1024.0, -1024.0, 3072.0)

    original_dcm = pydicom.dcmread(str(reference_path), force=True)
    slope = float(getattr(original_dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(original_dcm, "RescaleIntercept", 0.0))
    pixel_values = np.round((hu_values - intercept) / slope).astype(np.int16) if slope != 0 else np.round(hu_values).astype(np.int16)

    file_meta = original_dcm.file_meta
    if not getattr(file_meta, "TransferSyntaxUID", None):
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    new_dcm = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    for elem in original_dcm:
        if elem.tag != (0x7FE0, 0x0010):
            new_dcm.add(elem)
    new_dcm.PixelData = pixel_values.tobytes()
    new_dcm.Rows, new_dcm.Columns = pixel_values.shape
    new_dcm.save_as(str(output_path))


def load_mr_nifti(path: str | Path, image_size: int) -> np.ndarray:
    nii = nib.load(str(path))
    image = nii.get_fdata().astype(np.float32)
    if image.ndim == 3:
        image = image[:, :, image.shape[2] // 2]
    min_val = float(image.min())
    max_val = float(image.max())
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    image = _downsample_2d(image, image_size)
    return np.expand_dims(image.astype(np.float32), axis=0)


def save_mr_nifti(tensor: torch.Tensor, reference_path: str | Path, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.squeeze().detach().cpu().numpy().astype(np.float32)

    original_nii = nib.load(str(reference_path))
    original_data = original_nii.get_fdata().astype(np.float32)
    reference_slice = original_data[:, :, original_data.shape[2] // 2] if original_data.ndim == 3 else original_data
    min_val = float(reference_slice.min())
    max_val = float(reference_slice.max())
    if max_val > min_val:
        image = image * (max_val - min_val) + min_val

    orig_h, orig_w = reference_slice.shape
    new_h, new_w = image.shape
    step_h = max(1, orig_h // new_h)
    step_w = max(1, orig_w // new_w)
    scaling = np.diag([step_w, step_h, 1.0, 1.0])
    affine = original_nii.affine @ scaling
    nib.save(nib.Nifti1Image(image, affine), str(output_path))


def load_image(path: str | Path, image_size: int, io_format: str) -> np.ndarray:
    if io_format == "dicom":
        return load_ct_dicom(path, image_size)
    if io_format == "nifti":
        return load_mr_nifti(path, image_size)
    raise ValueError(f"Unsupported io_format: {io_format}")


def save_image(tensor: torch.Tensor, reference_path: str | Path, output_path: str | Path, io_format: str) -> None:
    if io_format == "dicom":
        save_ct_dicom(tensor, reference_path, output_path)
        return
    if io_format == "nifti":
        save_mr_nifti(tensor, reference_path, output_path)
        return
    raise ValueError(f"Unsupported io_format: {io_format}")


class GenericImageDataset(Dataset):
    def __init__(self, train_file: str | Path, image_size: int, io_format: str):
        self.paths = read_path_list(train_file)
        self.image_size = image_size
        self.io_format = io_format

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        target_path = self.paths[index]
        return torch.from_numpy(load_image(target_path, self.image_size, self.io_format))


def build_output_path(input_path: str, root_path: str | Path, output_dir: str | Path) -> Path:
    input_path = str(input_path)
    root_path = str(root_path)
    if root_path and input_path.startswith(root_path):
        relative = input_path[len(root_path):].lstrip("/\\")
    else:
        relative = Path(input_path).name
    return Path(output_dir) / relative


def batch_stack(images: Sequence[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(images, axis=0))
