from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


def ensure_ppu_runtime_path() -> None:
    sdk_root = Path("/usr/local/PPU_SDK")
    marker = "_FLOW_DIT_EVAL_PPU_ENV_READY"
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

from dit_zsq import DiT
from flow_matching_zsq import flow_interpolate


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "cifar10"
NORMALIZATION_MEAN = (0.5, 0.5, 0.5)
NORMALIZATION_STD = (0.5, 0.5, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Flow Matching checkpoint on its linear probability path."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--num-time-bins", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("training_objective") != "flow_matching":
        raise ValueError("The checkpoint is not a Flow Matching checkpoint.")

    model = DiT(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]
    )
    dataset = datasets.CIFAR10(
        root=args.data_dir, train=False, transform=transform, download=False
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
    counts = torch.zeros(args.num_time_bins, dtype=torch.long)
    velocity_mse_sums = torch.zeros(args.num_time_bins, dtype=torch.float64)
    noise_mse_sums = torch.zeros(args.num_time_bins, dtype=torch.float64)
    x0_mse_sums = torch.zeros(args.num_time_bins, dtype=torch.float64)
    clipped_x0_mse_sums = torch.zeros(args.num_time_bins, dtype=torch.float64)

    with torch.no_grad():
        for batch_index, (x0_cpu, labels_cpu) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch_size = x0_cpu.shape[0]
            t_cpu = torch.rand(batch_size, generator=generator, dtype=x0_cpu.dtype)
            noise_cpu = torch.randn(x0_cpu.shape, generator=generator, dtype=x0_cpu.dtype)

            x0 = x0_cpu.to(device, non_blocking=True)
            labels = labels_cpu.to(device, non_blocking=True)
            t = t_cpu.to(device, non_blocking=True)
            noise = noise_cpu.to(device, non_blocking=True)
            x_t, target_velocity = flow_interpolate(x0, t, noise)
            pred_velocity = model(x_t, t * 1000.0, labels)
            t_view = t.reshape(-1, *([1] * (x0.ndim - 1)))

            # From x_t=(1-t)*noise+t*x0 and v=x0-noise:
            # x0=x_t+(1-t)*v, noise=x_t-t*v.
            pred_x0 = x_t + (1.0 - t_view) * pred_velocity
            pred_noise = x_t - t_view * pred_velocity

            per_sample_velocity_mse = (
                (pred_velocity - target_velocity).square().flatten(1).mean(dim=1).cpu()
            )
            per_sample_noise_mse = (
                (pred_noise - noise).square().flatten(1).mean(dim=1).cpu()
            )
            per_sample_x0_mse = (
                (pred_x0 - x0).square().flatten(1).mean(dim=1).cpu()
            )
            per_sample_clipped_x0_mse = (
                (pred_x0.clamp(-1.0, 1.0) - x0).square().flatten(1).mean(dim=1).cpu()
            )
            bin_indices = torch.clamp(
                (t_cpu * args.num_time_bins).long(), max=args.num_time_bins - 1
            )
            for bin_index in range(args.num_time_bins):
                selected = bin_indices == bin_index
                selected_count = int(selected.sum())
                if selected_count == 0:
                    continue
                counts[bin_index] += selected_count
                velocity_mse_sums[bin_index] += per_sample_velocity_mse[selected].sum()
                noise_mse_sums[bin_index] += per_sample_noise_mse[selected].sum()
                x0_mse_sums[bin_index] += per_sample_x0_mse[selected].sum()
                clipped_x0_mse_sums[bin_index] += per_sample_clipped_x0_mse[selected].sum()

    total_count = int(counts.sum())
    overall_velocity_mse = float(velocity_mse_sums.sum() / total_count)
    overall_noise_mse = float(noise_mse_sums.sum() / total_count)
    overall_x0_mse = float(x0_mse_sums.sum() / total_count)
    overall_clipped_x0_mse = float(clipped_x0_mse_sums.sum() / total_count)
    results = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "training_objective": "flow_matching",
        "path": "x_t = t*x0 + (1-t)*noise",
        "seed": args.seed,
        "sample_count": total_count,
        "comparison_warning": (
            "Flow path metrics are internally comparable across t, but not numerically "
            "identical to diffusion-path metrics because the x_t distributions differ."
        ),
        "overall": {
            "velocity_mse": overall_velocity_mse,
            "noise_mse_on_flow_path": overall_noise_mse,
            "x0_mse_on_flow_path": overall_x0_mse,
            "clipped_x0_mse_on_flow_path": overall_clipped_x0_mse,
            "clipped_x0_psnr_on_flow_path": 10.0
            * math.log10(4.0 / overall_clipped_x0_mse),
        },
        "time_bins": [],
    }
    for bin_index in range(args.num_time_bins):
        count = int(counts[bin_index])
        clipped_mse = float(clipped_x0_mse_sums[bin_index] / count)
        results["time_bins"].append(
            {
                "t_start": bin_index / args.num_time_bins,
                "t_end": (bin_index + 1) / args.num_time_bins,
                "count": count,
                "velocity_mse": float(velocity_mse_sums[bin_index] / count),
                "noise_mse_on_flow_path": float(noise_mse_sums[bin_index] / count),
                "x0_mse_on_flow_path": float(x0_mse_sums[bin_index] / count),
                "clipped_x0_mse_on_flow_path": clipped_mse,
                "clipped_x0_psnr_on_flow_path": 10.0 * math.log10(4.0 / clipped_mse),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
