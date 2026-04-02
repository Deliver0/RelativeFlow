from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import UNetModel
from utils.data import GenericImageDataset


class DeltaSchedule:
    def __init__(self, delta_min: float, delta_max: float, alpha: float):
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.alpha = alpha

    def current(self) -> tuple[float, float]:
        return self.delta_min, self.delta_max

    def step(self, logger: logging.Logger) -> None:
        self.delta_min *= self.alpha
        self.delta_max /= self.alpha
        logger.info("updated delta_range=[%.4f, %.4f]", self.delta_min, self.delta_max)


MODALITY_DEFAULTS = {
    "ct": {
        "batch_size": 5,
        "output_dir": "weights/ct",
        "checkpoint_prefix": "ct",
        "logger_name": "train_ct",
    },
    "mr": {
        "batch_size": 20,
        "output_dir": "weights/mr",
        "checkpoint_prefix": "mr",
        "logger_name": "train_mr",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["ct", "mr"], required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--io-format", choices=["dicom", "nifti"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--delta-min-init", type=float, default=0.2)
    parser.add_argument("--delta-max-init", type=float, default=0.2)
    parser.add_argument("--delta-alpha", type=float, default=0.9)
    parser.add_argument("--target-clamp", type=float, default=10.0)
    return parser.parse_args(argv)


def build_logger(workspace: Path, log_dir: str, name: str) -> logging.Logger:
    workspace.joinpath(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(workspace / log_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def load_model(image_size: int, device: torch.device, resume: str) -> UNetModel:
    model = UNetModel(dim=(1, image_size, image_size), num_channels=64, out_channels=1, num_res_blocks=4, class_cond=False)
    model = model.to(device)
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("RelativeFlow_model", checkpoint))
        model.load_state_dict(state_dict)
    return model


def create_dataset(train_file: str, image_size: int, io_format: str) -> Dataset:
    return GenericImageDataset(train_file, image_size, io_format)


def create_simulator(modality: str, device: torch.device):
    from utils.simulation import DegradationSimulatorCT, DegradationSimulatorMR

    if modality == "ct":
        return DegradationSimulatorCT(device=device)
    return DegradationSimulatorMR(device=device)


def degrade_batch(simulator, modality: str, gt_images: torch.Tensor, min_delta: float, max_delta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    delta_scalar = torch.rand(1, device=device) * (max_delta - min_delta) + min_delta
    delta_t = delta_scalar.expand(gt_images.shape[0], 1, 1, 1)
    if modality == "ct":
        input_images = simulator.degrade(gt_images, float(delta_scalar.item()), ct_unit="unit")
        return input_images, delta_t
    input_images = simulator.degrade(gt_images, float(delta_scalar.item()))
    return input_images, delta_t


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    defaults = MODALITY_DEFAULTS[args.modality]
    batch_size = args.batch_size if args.batch_size is not None else defaults["batch_size"]
    output_dir_name = args.output_dir if args.output_dir is not None else defaults["output_dir"]
    logger_name = defaults["logger_name"]
    checkpoint_prefix = defaults["checkpoint_prefix"]

    workspace = Path(__file__).resolve().parent
    logger = build_logger(workspace, args.log_dir, logger_name)
    device = torch.device(args.device)
    dataset = create_dataset(args.train_file, args.image_size, args.io_format)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    model = load_model(args.image_size, device, args.resume)
    simulator = create_simulator(args.modality, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    delta_schedule = DeltaSchedule(args.delta_min_init, args.delta_max_init, args.delta_alpha)
    output_dir = workspace / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("modality=%s", args.modality)
    logger.info("io_format=%s", args.io_format)
    logger.info("train_file=%s", args.train_file)
    logger.info("dataset_size=%d", len(dataset))
    logger.info("batch_size=%d", batch_size)
    logger.info("loss=l2")
    logger.info("optimizer=Adam")
    logger.info("device=%s", device)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        min_delta, max_delta = delta_schedule.current()
        progress = tqdm(dataloader, dynamic_ncols=True, desc=f"train_{args.modality} epoch {epoch}")
        for gt_images in progress:
            gt_images = gt_images.to(device)
            optimizer.zero_grad()
            input_images, delta_t = degrade_batch(simulator, args.modality, gt_images, min_delta, max_delta, device)
            diff = gt_images - input_images
            denominator = torch.clamp(torch.exp(delta_t) - 1.0, min=1e-8)
            target_u = torch.clamp(diff / denominator, min=-args.target_clamp, max=args.target_clamp)
            pred_u = model(input_images, delta_t.view(-1))
            loss = F.mse_loss(pred_u, target_u)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("skip invalid loss at epoch=%d", epoch)
                continue
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{running_loss / max(1, progress.n):.4f}", delta_min=f"{min_delta:.3f}")
        epoch_loss = running_loss / max(1, len(dataloader))
        checkpoint_path = output_dir / f"{checkpoint_prefix}_epoch_{epoch + 1:03d}.pt"
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1, "loss": epoch_loss}, checkpoint_path)
        logger.info("epoch=%d loss=%.6f delta_range=[%.4f, %.4f]", epoch + 1, epoch_loss, min_delta, max_delta)
        logger.info("saved=%s", checkpoint_path)
        delta_schedule.step(logger)


if __name__ == "__main__":
    main()
