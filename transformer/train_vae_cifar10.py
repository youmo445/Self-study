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
    marker = "_VAE_PPU_ENV_READY"
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
        [
            Path("/opt/accl-p"),
            sdk_root / "lib",
            sdk_root / "targets" / "x86_64-linux" / "lib",
            sdk_root / "sailSHMEM" / "lib",
        ],
    )
    environment["LIBRARY_PATH"] = prepend_paths(
        "LIBRARY_PATH",
        [sdk_root / "lib"],
    )
    environment["PYTHONUNBUFFERED"] = "1"
    environment[marker] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


if __name__ == "__main__":
    ensure_ppu_runtime_path()


import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image

from vae_zsq import VAE, vae_loss, vae_train_step


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "cifar10"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "checkpoints" / "vae_cifar10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hand-written VAE on CIFAR-10.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--visualize-every", type=int, default=5)
    parser.add_argument("--visualize-count", type=int, default=16)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-download", action="store_true")

    # These limits are zero during real training. They only make a quick smoke test cheap.
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-validation-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)

    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    download = not args.no_download
    train_augmented = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        transform=train_transform,
        download=download,
    )
    train_for_validation = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        transform=evaluation_transform,
        download=False,
    )
    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=evaluation_transform,
        download=False,
    )

    if not 0 < args.validation_size < len(train_augmented):
        raise ValueError("validation-size must be between 1 and len(CIFAR-10 train)-1")

    split_generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(train_augmented), generator=split_generator).tolist()
    validation_indices = permutation[: args.validation_size]
    train_indices = permutation[args.validation_size :]

    train_dataset = Subset(train_augmented, train_indices)
    validation_dataset = Subset(train_for_validation, validation_indices)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, validation_loader, test_loader


def current_kl_weight(
    target_weight: float,
    warmup_epochs: int,
    epoch: int,
    step: int,
    steps_per_epoch: int,
) -> float:
    if warmup_epochs <= 0:
        return target_weight
    completed_epochs = (epoch - 1) + step / steps_per_epoch
    fraction = min(1.0, completed_epochs / warmup_epochs)
    return target_weight * fraction


def train_one_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    writer: SummaryWriter,
    args: argparse.Namespace,
) -> tuple[dict[str, float], int]:
    model.train()
    total_sum = 0.0
    reconstruction_sum = 0.0
    kl_sum = 0.0
    sample_count = 0
    epoch_start = time.time()

    for step, (images, _) in enumerate(loader, start=1):
        if args.max_train_batches > 0 and step > args.max_train_batches:
            break

        images = images.to(device, non_blocking=True)
        kl_weight = current_kl_weight(
            args.kl_weight,
            args.kl_warmup_epochs,
            epoch,
            step,
            len(loader),
        )
        total, reconstruction, kl = vae_train_step(
            model,
            images,
            optimizer,
            kl_weight=kl_weight,
        )

        batch_size = images.shape[0]
        total_sum += total * batch_size
        reconstruction_sum += reconstruction * batch_size
        kl_sum += kl * batch_size
        sample_count += batch_size
        global_step += 1

        writer.add_scalar("train/total_loss_step", total, global_step)
        writer.add_scalar("train/reconstruction_loss_step", reconstruction, global_step)
        writer.add_scalar("train/kl_loss_step", kl, global_step)
        writer.add_scalar("train/kl_weight_step", kl_weight, global_step)

        if step % args.log_every == 0 or step == len(loader):
            elapsed = time.time() - epoch_start
            print(
                f"epoch={epoch:03d} step={step:04d}/{len(loader):04d} "
                f"loss={total_sum / sample_count:.4f} "
                f"reconstruction={reconstruction_sum / sample_count:.4f} "
                f"kl={kl_sum / sample_count:.4f} "
                f"beta={kl_weight:.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    if sample_count == 0:
        raise RuntimeError("the training loader produced no samples")

    return (
        {
            "total_loss": total_sum / sample_count,
            "reconstruction_loss": reconstruction_sum / sample_count,
            "kl_loss": kl_sum / sample_count,
        },
        global_step,
    )


@torch.no_grad()
def evaluate(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    kl_weight: float,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    total_sum = 0.0
    reconstruction_sum = 0.0
    kl_sum = 0.0
    mu_mean_sum = 0.0
    mu_std_sum = 0.0
    posterior_std_sum = 0.0
    sample_count = 0

    for batch_index, (images, _) in enumerate(loader, start=1):
        if max_batches > 0 and batch_index > max_batches:
            break

        images = images.to(device, non_blocking=True)
        mu, logvar = model.encoder(images)

        # z=mu removes sampling noise, so validation curves are directly comparable.
        reconstruction = model.decode(mu)
        total, reconstruction_loss, kl_loss = vae_loss(
            reconstruction,
            images,
            mu,
            logvar,
            kl_weight=kl_weight,
        )

        batch_size = images.shape[0]
        total_sum += total.item() * batch_size
        reconstruction_sum += reconstruction_loss.item() * batch_size
        kl_sum += kl_loss.item() * batch_size
        mu_mean_sum += mu.mean().item() * batch_size
        mu_std_sum += mu.std(unbiased=False).item() * batch_size
        posterior_std_sum += torch.exp(0.5 * logvar).mean().item() * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("the evaluation loader produced no samples")

    return {
        "total_loss": total_sum / sample_count,
        "reconstruction_loss": reconstruction_sum / sample_count,
        "kl_loss": kl_sum / sample_count,
        "mu_mean": mu_mean_sum / sample_count,
        "mu_std": mu_std_sum / sample_count,
        "posterior_std_mean": posterior_std_sum / sample_count,
    }


def to_image_range(images: torch.Tensor) -> torch.Tensor:
    return ((images.detach().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)


@torch.no_grad()
def save_visuals(
    model: VAE,
    fixed_images: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    writer: SummaryWriter,
    tag: str,
    tensorboard_step: int,
) -> None:
    model.eval()
    images = fixed_images.to(device)
    mu, _ = model.encoder(images)
    reconstructions = model.decode(mu)

    prior_latents = torch.randn_like(mu)
    prior_samples = model.decode(prior_latents)

    images_for_display = to_image_range(images)
    reconstructions_for_display = to_image_range(reconstructions)
    samples_for_display = to_image_range(prior_samples)

    # With nrow=N, originals form the first row and reconstructions the second row.
    comparison = torch.cat([images_for_display, reconstructions_for_display], dim=0)
    save_image(
        comparison,
        output_dir / "reconstructions" / f"reconstruction_{tag}.png",
        nrow=images_for_display.shape[0],
    )
    sample_nrow = max(1, int(math.sqrt(samples_for_display.shape[0])))
    save_image(
        samples_for_display,
        output_dir / "samples" / f"prior_samples_{tag}.png",
        nrow=sample_nrow,
    )

    writer.add_images("images/original", images_for_display, tensorboard_step)
    writer.add_images("images/reconstruction_from_mu", reconstructions_for_display, tensorboard_step)
    writer.add_images("images/prior_samples", samples_for_display, tensorboard_step)


def save_checkpoint(path: Path, state: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def json_ready_args(args: argparse.Namespace) -> dict:
    result = vars(args).copy()
    for key, value in result.items():
        if isinstance(value, Path):
            result[key] = str(value)
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA/PPU is unavailable, but a cuda device was requested")
    device = torch.device(args.device)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reconstructions").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "samples").mkdir(parents=True, exist_ok=True)
    tensorboard_dir = args.output_dir / "tensorboard"
    writer = SummaryWriter(tensorboard_dir)

    train_loader, validation_loader, test_loader = build_dataloaders(args, device)
    model_config = {
        "in_channels": 3,
        "base_channels": args.base_channels,
        "latent_channels": args.latent_channels,
    }
    model = VAE(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_learning_rate,
    )

    start_epoch = 1
    global_step = 0
    best_validation_loss = math.inf
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_validation_loss = float(checkpoint.get("best_validation_loss", math.inf))
        print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_config = {
        **json_ready_args(args),
        "model_config": model_config,
        "parameter_count": parameter_count,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "input_range": [-1.0, 1.0],
        "split_seed": args.seed,
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(run_config, file, ensure_ascii=False, indent=2)

    fixed_images, _ = next(iter(validation_loader))
    fixed_images = fixed_images[: args.visualize_count]

    print(f"device={device}", flush=True)
    print(f"parameters={parameter_count:,}", flush=True)
    print(
        f"samples: train={len(train_loader.dataset)} "
        f"validation={len(validation_loader.dataset)} test={len(test_loader.dataset)}",
        flush=True,
    )
    print(f"tensorboard={tensorboard_dir}", flush=True)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics, global_step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                global_step,
                writer,
                args,
            )
            validation_metrics = evaluate(
                model,
                validation_loader,
                device,
                kl_weight=args.kl_weight,
                max_batches=args.max_validation_batches,
            )
            scheduler.step()

            for metric_name, value in train_metrics.items():
                writer.add_scalar(f"train/{metric_name}_epoch", value, epoch)
            for metric_name, value in validation_metrics.items():
                writer.add_scalar(f"validation/{metric_name}", value, epoch)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)

            should_visualize = (
                epoch == 1
                or epoch == args.epochs
                or (args.visualize_every > 0 and epoch % args.visualize_every == 0)
            )
            if should_visualize:
                save_visuals(
                    model,
                    fixed_images,
                    device,
                    args.output_dir,
                    writer,
                    tag=f"epoch_{epoch:04d}",
                    tensorboard_step=epoch,
                )

            checkpoint_state = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_validation_loss": min(
                    best_validation_loss,
                    validation_metrics["total_loss"],
                ),
                "model_config": model_config,
                "args": json_ready_args(args),
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
            }
            save_checkpoint(args.output_dir / "latest.pt", checkpoint_state)

            is_best = validation_metrics["total_loss"] < best_validation_loss
            if is_best:
                best_validation_loss = validation_metrics["total_loss"]
                checkpoint_state["best_validation_loss"] = best_validation_loss
                save_checkpoint(args.output_dir / "best.pt", checkpoint_state)

            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_metrics['total_loss']:.4f} "
                f"val_loss={validation_metrics['total_loss']:.4f} "
                f"val_reconstruction={validation_metrics['reconstruction_loss']:.4f} "
                f"val_kl={validation_metrics['kl_loss']:.4f} "
                f"mu_mean={validation_metrics['mu_mean']:.4f} "
                f"mu_std={validation_metrics['mu_std']:.4f} "
                f"posterior_std={validation_metrics['posterior_std_mean']:.4f} "
                f"best={best_validation_loss:.4f}"
                f"{' *' if is_best else ''}",
                flush=True,
            )

        best_path = args.output_dir / "best.pt"
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            kl_weight=args.kl_weight,
            max_batches=args.max_test_batches,
        )
        for metric_name, value in test_metrics.items():
            writer.add_scalar(f"test/{metric_name}", value, args.epochs)

        save_visuals(
            model,
            fixed_images,
            device,
            args.output_dir,
            writer,
            tag="best",
            tensorboard_step=args.epochs,
        )
        with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(test_metrics, file, ensure_ascii=False, indent=2)
        print(
            "test: " + " ".join(f"{key}={value:.4f}" for key, value in test_metrics.items()),
            flush=True,
        )
        print(f"best_checkpoint={best_path}", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
