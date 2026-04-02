from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models import UNetModel
from utils.data import build_output_path, load_image, read_paired_paths, save_image


MODALITY_DEFAULTS = {
    "ct": {
        "batch_size": 8,
        "results_dir": "outputs/ct/testA",
        "logger_name": "predict_ct",
    },
    "mr": {
        "batch_size": 4,
        "results_dir": "outputs/mr/testA",
        "logger_name": "predict_mr",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["ct", "mr"], required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--io-format", choices=["dicom", "nifti"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--delta-seq", type=str, default="0.2,0.1,0.05")
    parser.add_argument("--update-factor", choices=["one_minus_exp_neg", "expm1", "identity"], default="identity")
    parser.add_argument("--results-dir", type=str, default=None)
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


def parse_sequence(sequence: str) -> list[float]:
    return [max(float(item.strip()), 1e-8) for item in sequence.split(",") if item.strip()]


def compute_factor(delta_t: torch.Tensor, rule: str) -> torch.Tensor:
    if rule == "expm1":
        return torch.expm1(delta_t)
    if rule == "identity":
        return delta_t
    return 1.0 - torch.exp(-delta_t)


def load_model(image_size: int, ckpt_path: str, device: torch.device) -> UNetModel:
    model = UNetModel(dim=(1, image_size, image_size), num_channels=64, out_channels=1, num_res_blocks=4, class_cond=False)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("RelativeFlow_model", checkpoint))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def run_batch(model: UNetModel, batch_input: torch.Tensor, delta_seq: list[float], rule: str, device: torch.device) -> torch.Tensor:
    current_x = batch_input.to(device)
    with torch.no_grad():
        for delta_value in delta_seq:
            delta_t = torch.full((current_x.shape[0],), float(delta_value), device=device)
            pred_u = model(current_x, delta_t)
            factor = compute_factor(delta_t, rule).view(-1, 1, 1, 1)
            current_x = current_x + pred_u * factor
        return torch.clamp(current_x, 0.0, 1.0)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    defaults = MODALITY_DEFAULTS[args.modality]
    batch_size = args.batch_size if args.batch_size is not None else defaults["batch_size"]
    results_dir_name = args.results_dir if args.results_dir is not None else defaults["results_dir"]
    logger_name = defaults["logger_name"]

    workspace = Path(__file__).resolve().parent
    logger = build_logger(workspace, args.log_dir, logger_name)
    device = torch.device(args.device)
    delta_seq = parse_sequence(args.delta_seq)
    model = load_model(args.image_size, args.ckpt, device)
    pairs = read_paired_paths(args.test_file)
    logger.info("modality=%s", args.modality)
    logger.info("io_format=%s", args.io_format)
    logger.info("test_pairs=%d", len(pairs))
    logger.info("delta_seq=%s", delta_seq)
    results_dir = workspace / results_dir_name
    for start in tqdm(range(0, len(pairs), batch_size), desc=f"predict_{args.modality}", dynamic_ncols=True):
        batch_pairs = pairs[start:start + batch_size]
        batch_input_paths = [pair[0] for pair in batch_pairs]
        batch_images = [load_image(path, args.image_size, args.io_format) for path in batch_input_paths]
        batch_tensor = torch.from_numpy(np.stack(batch_images, axis=0))
        batch_output = run_batch(model, batch_tensor, delta_seq, args.update_factor, device)
        for index, input_path in enumerate(batch_input_paths):
            output_path = build_output_path(input_path, args.root_path, results_dir)
            save_image(batch_output[index:index + 1], input_path, output_path, args.io_format)
    logger.info("results_dir=%s", results_dir)


if __name__ == "__main__":
    main()
