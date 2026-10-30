from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path


def ensure_ppu_runtime_path() -> None:
    """Restart once with the PPU runtime configured before importing torch."""
    sdk_root = Path("/usr/local/PPU_SDK")
    marker = "_FLOW_DIT_PPU_ENV_READY"
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
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

import train_dit_cifar10 as common
from dit_zsq import DiT
from flow_matching_zsq import flow_matching_loss


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "cifar10"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "checkpoints" / "dit_compare_flow_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the hand-written class-conditional DiT with Flow Matching."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-validation-steps", type=int, default=None)

    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: DiT,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    writer: SummaryWriter,
    args: argparse.Namespace,
) -> tuple[float, int]:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    epoch_start = time.time()
    total_steps = len(loader)
    if args.max_train_steps is not None:
        total_steps = min(total_steps, args.max_train_steps)

    for step, (images, labels) in enumerate(loader, start=1):
        if step > total_steps:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # Continuous Flow Matching time: one independently sampled t in [0,1) per image.
        t = torch.rand(images.shape[0], device=device, dtype=images.dtype)

        optimizer.zero_grad(set_to_none=True)
        loss = flow_matching_loss(model, images, t, labels)
        loss.backward()

        if args.grad_clip > 0:
            grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_norm_value = float(grad_norm)
        else:
            grad_norm_value = math.nan
        optimizer.step()

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        global_step += 1
        writer.add_scalar("train/loss_step", loss.item(), global_step)
        if math.isfinite(grad_norm_value):
            writer.add_scalar("train/grad_norm_step", grad_norm_value, global_step)

        if step % args.log_every == 0 or step == total_steps:
            elapsed = time.time() - epoch_start
            print(
                f"epoch={epoch:03d} step={step:04d}/{total_steps:04d} "
                f"loss={loss_sum / sample_count:.6f} "
                f"samples_per_second={sample_count / elapsed:.1f}",
                flush=True,
            )

    return loss_sum / sample_count, global_step


@torch.no_grad()
def evaluate(
    model: DiT,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    evaluation_seed: int,
    max_steps: int | None,
) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    # CPU RNG makes every epoch see exactly the same (t, noise) validation pairs.
    generator = torch.Generator().manual_seed(evaluation_seed)
    total_steps = len(loader) if max_steps is None else min(len(loader), max_steps)

    for step, (images, labels) in enumerate(loader, start=1):
        if step > total_steps:
            break
        batch_size = images.shape[0]
        t = torch.rand(batch_size, generator=generator, dtype=images.dtype)
        noise = torch.randn(images.shape, generator=generator, dtype=images.dtype)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        noise = noise.to(device, non_blocking=True)
        loss = flow_matching_loss(model, images, t, labels, noise)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size

    return loss_sum / sample_count


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA/PPU is not available, but a cuda device was requested.")
    device = torch.device(args.device)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = args.output_dir / "tensorboard"
    writer = SummaryWriter(tensorboard_dir)

    train_loader, validation_loader, test_loader, class_names = common.build_dataloaders(
        args, device
    )
    model, model_config = common.build_model(args)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_learning_rate
    )

    flow_config = {
        "path": "x_t = t * x_data + (1 - t) * noise",
        "target_velocity": "x_data - noise",
        "time_scale_for_dit": 1000.0,
        "solver": "forward_euler",
    }
    start_epoch = 1
    global_step = 0
    best_validation_loss = math.inf

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["model_config"] != model_config:
            raise ValueError("The checkpoint model_config does not match the requested model.")
        if checkpoint.get("training_objective") != "flow_matching":
            raise ValueError("The checkpoint is not a Flow Matching checkpoint.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        common.move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_validation_loss = float(checkpoint.get("best_validation_loss", math.inf))
        print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_config = {
        **vars(args),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "resume": None if args.resume is None else str(args.resume),
        "training_objective": "flow_matching",
        "model_config": model_config,
        "flow_config": flow_config,
        "parameter_count": parameter_count,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "class_names": class_names,
        "normalization_mean": common.DIFFUSION_MEAN,
        "normalization_std": common.DIFFUSION_STD,
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(run_config, file, ensure_ascii=False, indent=2)

    print(f"device={device}", flush=True)
    print(f"parameters={parameter_count:,}", flush=True)
    print(
        f"samples: train={len(train_loader.dataset)} "
        f"validation={len(validation_loader.dataset)} test={len(test_loader.dataset)}",
        flush=True,
    )
    print("training_objective=flow_matching", flush=True)
    print(f"tensorboard={tensorboard_dir}", flush=True)

    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, global_step = train_one_epoch(
                model, train_loader, optimizer, device, epoch, global_step, writer, args
            )
            validation_loss = evaluate(
                model,
                validation_loader,
                device,
                args.seed + 10_000,
                args.max_validation_steps,
            )
            scheduler.step()
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("validation/loss", validation_loss, epoch)
            writer.add_scalar("optimization/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            writer.flush()

            is_best = validation_loss < best_validation_loss
            if is_best:
                best_validation_loss = validation_loss
            state = {
                "epoch": epoch,
                "global_step": global_step,
                "best_validation_loss": best_validation_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "model_config": model_config,
                "training_objective": "flow_matching",
                "flow_config": flow_config,
                "class_names": class_names,
                "normalization_mean": common.DIFFUSION_MEAN,
                "normalization_std": common.DIFFUSION_STD,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
            common.save_checkpoint(latest_path, state)
            if is_best:
                common.save_checkpoint(best_path, state)
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"val_loss={validation_loss:.6f} "
                f"best_val_loss={best_validation_loss:.6f}",
                flush=True,
            )

        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        model.to(device)
        test_loss = evaluate(
            model, test_loader, device, args.seed + 20_000, args.max_validation_steps
        )
        writer.add_scalar("test/loss", test_loss, 0)
        writer.flush()
        print(
            f"best_epoch={best_checkpoint['epoch']} test_loss={test_loss:.6f}",
            flush=True,
        )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
