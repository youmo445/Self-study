from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


def ensure_ppu_runtime_path() -> None:
    sdk_root = Path("/usr/local/PPU_SDK")
    marker = "_DIT_EVAL_PPU_ENV_READY"
    if not sdk_root.is_dir() or os.environ.get(marker) == "1":
        return

    def prepend_paths(variable: str, new_paths: list[Path]) -> str:
        old_paths = [path for path in os.environ.get(variable, "").split(":") if path]
        combined = [str(path) for path in new_paths if path.is_dir()]
        combined.extend(old_paths)
        return ":".join(dict.fromkeys(combined))

    environment = os.environ.copy()
    environment["PPU_SDK"] = str(sdk_root)
    environment["PPU_PATH"] = str(sdk_root)
    environment["PPU_HOME"] = str(sdk_root)
    environment["SAILSHMEM_DIR"] = str(sdk_root / "sailSHMEM")
    environment["PATH"] = prepend_paths(
        "PATH",
        [sdk_root / "bin", sdk_root / "asight" / "bin", sdk_root / "ppu-smi" / "bin"],
    )
    environment["LD_LIBRARY_PATH"] = prepend_paths(
        "LD_LIBRARY_PATH",
        [Path("/opt/accl-p"), sdk_root / "lib", sdk_root / "sailSHMEM" / "lib"],
    )
    environment["LIBRARY_PATH"] = prepend_paths("LIBRARY_PATH", [sdk_root / "lib"])
    environment["PYTHONUNBUFFERED"] = "1"
    environment[marker] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


if __name__ == "__main__":
    ensure_ppu_runtime_path()


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from diffusion_zsq import (
    make_beta_schedule,
    model_output_to_noise_and_x0,
    precompute_diffusion_items,
    q_sample,
)
from dit_zsq import DiT


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "cifar10"
NORMALIZATION_MEAN = (0.5, 0.5, 0.5)
NORMALIZATION_STD = (0.5, 0.5, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate epsilon/v checkpoints in a shared prediction space."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--num-timestep-bins", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    prediction_type = checkpoint.get("prediction_type", "epsilon")
    model_config = checkpoint["model_config"]
    diffusion_config = checkpoint["diffusion_config"]

    model = DiT(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    betas = make_beta_schedule(
        diffusion_config["num_diffusion_steps"],
        diffusion_config["beta_start"],
        diffusion_config["beta_end"],
        device=device,
    )
    _, _, sqrt_alpha_bars, sqrt_one_minus_alpha_bars = precompute_diffusion_items(
        betas
    )
    num_diffusion_steps = int(diffusion_config["num_diffusion_steps"])

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]
    )
    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=transform,
        download=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    generator = torch.Generator().manual_seed(args.seed)
    num_bins = args.num_timestep_bins
    bin_width = math.ceil(num_diffusion_steps / num_bins)
    counts = torch.zeros(num_bins, dtype=torch.long)
    noise_mse_sums = torch.zeros(num_bins, dtype=torch.float64)
    x0_mse_sums = torch.zeros(num_bins, dtype=torch.float64)
    clipped_x0_mse_sums = torch.zeros(num_bins, dtype=torch.float64)

    with torch.no_grad():
        for batch_index, (x0_cpu, labels_cpu) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break

            batch_size = x0_cpu.shape[0]
            t_cpu = torch.randint(
                0,
                num_diffusion_steps,
                (batch_size,),
                generator=generator,
                dtype=torch.long,
            )
            noise_cpu = torch.randn(
                x0_cpu.shape,
                generator=generator,
                dtype=x0_cpu.dtype,
            )

            x0 = x0_cpu.to(device, non_blocking=True)
            labels = labels_cpu.to(device, non_blocking=True)
            t = t_cpu.to(device, non_blocking=True)
            noise = noise_cpu.to(device, non_blocking=True)
            x_t, _ = q_sample(
                x0,
                t,
                sqrt_alpha_bars,
                sqrt_one_minus_alpha_bars,
                noise,
            )
            model_output = model(x_t, t, labels)
            pred_noise, pred_x0 = model_output_to_noise_and_x0(
                x_t,
                t,
                model_output,
                sqrt_alpha_bars,
                sqrt_one_minus_alpha_bars,
                prediction_type,
            )

            per_sample_noise_mse = (
                (pred_noise - noise).square().flatten(1).mean(dim=1).cpu()
            )
            per_sample_x0_mse = (
                (pred_x0 - x0).square().flatten(1).mean(dim=1).cpu()
            )
            per_sample_clipped_x0_mse = (
                (pred_x0.clamp(-1.0, 1.0) - x0)
                .square()
                .flatten(1)
                .mean(dim=1)
                .cpu()
            )
            bin_indices = torch.clamp(t_cpu // bin_width, max=num_bins - 1)

            for bin_index in range(num_bins):
                selected = bin_indices == bin_index
                selected_count = int(selected.sum())
                if selected_count == 0:
                    continue
                counts[bin_index] += selected_count
                noise_mse_sums[bin_index] += per_sample_noise_mse[selected].sum()
                x0_mse_sums[bin_index] += per_sample_x0_mse[selected].sum()
                clipped_x0_mse_sums[bin_index] += per_sample_clipped_x0_mse[
                    selected
                ].sum()

    total_count = int(counts.sum())
    overall_noise_mse = float(noise_mse_sums.sum() / total_count)
    overall_x0_mse = float(x0_mse_sums.sum() / total_count)
    overall_clipped_x0_mse = float(clipped_x0_mse_sums.sum() / total_count)
    results = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "prediction_type": prediction_type,
        "seed": args.seed,
        "sample_count": total_count,
        "overall": {
            "noise_mse": overall_noise_mse,
            "x0_mse": overall_x0_mse,
            "clipped_x0_mse": overall_clipped_x0_mse,
            "clipped_x0_psnr": 10.0
            * math.log10(4.0 / overall_clipped_x0_mse),
        },
        "timestep_bins": [],
    }
    for bin_index in range(num_bins):
        count = int(counts[bin_index])
        start = bin_index * bin_width
        end = min((bin_index + 1) * bin_width - 1, num_diffusion_steps - 1)
        clipped_mse = float(clipped_x0_mse_sums[bin_index] / count)
        results["timestep_bins"].append(
            {
                "t_start": start,
                "t_end": end,
                "count": count,
                "noise_mse": float(noise_mse_sums[bin_index] / count),
                "x0_mse": float(x0_mse_sums[bin_index] / count),
                "clipped_x0_mse": clipped_mse,
                "clipped_x0_psnr": 10.0 * math.log10(4.0 / clipped_mse),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
