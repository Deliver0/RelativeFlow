from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from utils import MetricComputer
from utils.data import build_output_path, load_mr_nifti, read_paired_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--predictions-dir", type=str, default="outputs/mr/testA")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--root-path", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="logs")
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


def main() -> None:
    args = parse_args()
    workspace = Path(__file__).resolve().parent
    logger = build_logger(workspace, args.log_dir, "evaluate_mr")
    device = torch.device(args.device)
    metric = MetricComputer(device=args.device)
    pairs = read_paired_paths(args.test_file)
    predictions_dir = workspace / args.predictions_dir
    csv_path = workspace / args.log_dir / f"evaluate_mr_{datetime.now():%Y%m%d_%H%M%S}.csv"
    totals = {"psnr": 0.0, "ssim": 0.0, "rmse": 0.0, "lpips": 0.0}
    count = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input_path", "target_path", "prediction_path", "psnr", "ssim", "rmse", "lpips"])
        for input_path, target_path in tqdm(pairs, desc="evaluate_mr", dynamic_ncols=True):
            prediction_path = build_output_path(input_path, args.root_path, predictions_dir)
            if not prediction_path.exists():
                logger.warning("missing prediction: %s", prediction_path)
                continue
            prediction = torch.from_numpy(load_mr_nifti(prediction_path, args.image_size)).unsqueeze(0).to(device)
            target = torch.from_numpy(load_mr_nifti(target_path, args.image_size)).unsqueeze(0).to(device)
            psnr, ssim, rmse, lpips = metric.compute_all(prediction, target, mode="MR")
            writer.writerow([input_path, target_path, str(prediction_path), psnr, ssim, rmse, lpips])
            totals["psnr"] += psnr
            totals["ssim"] += ssim
            totals["rmse"] += rmse
            totals["lpips"] += lpips
            count += 1
    if count > 0:
        logger.info("psnr=%.4f ssim=%.4f rmse=%.4f lpips=%.4f", totals["psnr"] / count, totals["ssim"] / count, totals["rmse"] / count, totals["lpips"] / count)
    logger.info("csv=%s", csv_path)


if __name__ == "__main__":
    main()
