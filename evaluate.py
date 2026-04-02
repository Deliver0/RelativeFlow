from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from utils.data import build_output_path, load_image, read_paired_paths


MODALITY_DEFAULTS = {
    "ct": {
        "predictions_dir": "outputs/ct/testA",
        "logger_name": "evaluate_ct",
        "metric_mode": "CT",
    },
    "mr": {
        "predictions_dir": "outputs/mr/testA",
        "logger_name": "evaluate_mr",
        "metric_mode": "MR",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["ct", "mr"], required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--predictions-dir", type=str, default=None)
    parser.add_argument("--io-format", choices=["dicom", "nifti"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--root-path", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="logs")
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from utils.metrics import MetricComputer

    defaults = MODALITY_DEFAULTS[args.modality]
    predictions_dir_name = args.predictions_dir if args.predictions_dir is not None else defaults["predictions_dir"]
    logger_name = defaults["logger_name"]
    metric_mode = defaults["metric_mode"]

    workspace = Path(__file__).resolve().parent
    logger = build_logger(workspace, args.log_dir, logger_name)
    device = torch.device(args.device)
    metric = MetricComputer(device=args.device)
    pairs = read_paired_paths(args.test_file)
    logger.info("modality=%s", args.modality)
    logger.info("io_format=%s", args.io_format)
    predictions_dir = workspace / predictions_dir_name
    csv_path = workspace / args.log_dir / f"{logger_name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    totals = {"psnr": 0.0, "ssim": 0.0, "rmse": 0.0, "lpips": 0.0}
    count = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input_path", "target_path", "prediction_path", "psnr", "ssim", "rmse", "lpips"])
        for input_path, target_path in tqdm(pairs, desc=f"evaluate_{args.modality}", dynamic_ncols=True):
            prediction_path = build_output_path(input_path, args.root_path, predictions_dir)
            if not prediction_path.exists():
                logger.warning("missing prediction: %s", prediction_path)
                continue
            prediction = torch.from_numpy(load_image(prediction_path, args.image_size, args.io_format)).unsqueeze(0).to(device)
            target = torch.from_numpy(load_image(target_path, args.image_size, args.io_format)).unsqueeze(0).to(device)
            psnr, ssim, rmse, lpips = metric.compute_all(prediction, target, mode=metric_mode)
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
