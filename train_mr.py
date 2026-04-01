from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import schedulefree
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import UNetModel
from utils import DegradationSimulatorMR, MRDataset


@dataclass
class AdaptiveDeltaRange:
    current_min: float
    floor: float
    delta_max: float
    step: float
    patience_limit: int
    best_loss: float = float("inf")
    patience_counter: int = 0

    def current(self) -> tuple[float, float]:
        return self.current_min, self.delta_max

    def update(self, epoch_loss: float, logger: logging.Logger) -> None:
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.patience_counter = 0
            if self.current_min > self.floor:
                self.current_min = max(self.current_min - self.step, self.floor)
                logger.info("updated delta_min=%.4f", self.current_min)
            return
        self.patience_counter += 1
        if self.patience_counter > self.patience_limit:
            self.patience_counter = self.patience_limit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--loss", choices=["l1", "l2"], default="l2")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="weights/mr")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--delta-min-init", type=float, default=0.1)
    parser.add_argument("--delta-min-floor", type=float, default=0.02)
    parser.add_argument("--delta-max", type=float, default=1.0)
    parser.add_argument("--delta-min-step", type=float, default=0.005)
    parser.add_argument("--delta-patience", type=int, default=2)
    parser.add_argument("--target-clamp", type=float, default=10.0)
    return parser.parse_args()


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


def compute_loss(prediction: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "l2":
        return F.mse_loss(prediction, target)
    return F.l1_loss(prediction, target)


def main() -> None:
    args = parse_args()
    workspace = Path(__file__).resolve().parent
    logger = build_logger(workspace, args.log_dir, "train_mr")
    device = torch.device(args.device)
    dataset = MRDataset(args.train_file, args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    model = load_model(args.image_size, device, args.resume)
    simulator = DegradationSimulatorMR(device=device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.lr)
    optimizer.train()
    delta_range = AdaptiveDeltaRange(
        current_min=args.delta_min_init,
        floor=args.delta_min_floor,
        delta_max=args.delta_max,
        step=args.delta_min_step,
        patience_limit=args.delta_patience,
    )
    output_dir = workspace / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("train_file=%s", args.train_file)
    logger.info("dataset_size=%d", len(dataset))
    logger.info("loss=%s", args.loss)
    logger.info("device=%s", device)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        min_delta, max_delta = delta_range.current()
        progress = tqdm(dataloader, dynamic_ncols=True, desc=f"train_mr epoch {epoch}")
        for gt_images in progress:
            gt_images = gt_images.to(device)
            optimizer.zero_grad()
            delta_scalar = torch.rand(1, device=device) * (max_delta - min_delta) + min_delta
            delta_t = delta_scalar.expand(gt_images.shape[0], 1, 1, 1)
            input_images = simulator.degrade(gt_images, float(delta_scalar.item()))
            diff = gt_images - input_images
            denominator = torch.clamp(torch.exp(delta_t) - 1.0, min=1e-8)
            target_u = torch.clamp(diff / denominator, min=-args.target_clamp, max=args.target_clamp)
            pred_u = model(input_images, delta_t.view(-1))
            loss = compute_loss(pred_u, target_u, args.loss)
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
        delta_range.update(epoch_loss, logger)
        checkpoint_path = output_dir / f"mr_epoch_{epoch + 1:03d}.pt"
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1, "loss": epoch_loss}, checkpoint_path)
        logger.info("epoch=%d loss=%.6f delta_range=[%.4f, %.4f]", epoch + 1, epoch_loss, min_delta, max_delta)
        logger.info("saved=%s", checkpoint_path)


if __name__ == "__main__":
    main()
