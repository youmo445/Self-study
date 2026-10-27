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
    """Restart once with the PPU SDK runtime configured before importing torch."""
    sdk_root = Path("/usr/local/PPU_SDK")
    marker = "_VIT_PPU_ENV_READY"
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
        [Path("/opt/accl-p"), sdk_root / "lib"],
    )
    environment["LIBRARY_PATH"] = prepend_paths(
        "LIBRARY_PATH",
        [sdk_root / "lib"],
    environment["PYTHONUNBUFFERED"] = "1"
    environment[marker] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


if __name__ == "__main__":
    ensure_ppu_runtime_path()


import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from vit_zsq import VisionTransformer


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "cifar10"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "checkpoints" / "vit_cifar10"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hand-written ViT on CIFAR-10.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", type=Path, default=None)

    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(args: argparse.Namespace, device: torch.device):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    # The first call downloads/extracts both CIFAR-10 train and test batches.
    train_augmented = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        transform=train_transform,
        download=True,
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

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(train_augmented), generator=generator).tolist()
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
    return train_loader, validation_loader, test_loader, train_augmented.classes


def build_model(args: argparse.Namespace) -> tuple[VisionTransformer, dict[str, int | float]]:
    model_config: dict[str, int | float] = {
        "H": 32,
        "W": 32,
        "in_channels": 3,
        "patch_size": args.patch_size,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "mlp_dim": args.mlp_dim,
        "num_layers": args.num_layers,
        "num_classes": 10,
        "dropout": args.dropout,
    }
    return VisionTransformer(**model_config), model_config


def train_one_epoch(
    model: VisionTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    writer: SummaryWriter,
    args: argparse.Namespace,
) -> tuple[float, float, int]:
    model.train()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    epoch_start = time.time()

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
        loss.backward()
        if args.grad_clip > 0:
            clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        batch_size = labels.shape[0]
        loss_sum += loss.item() * batch_size
        correct += logits.argmax(dim=-1).eq(labels).sum().item()
        sample_count += batch_size
        global_step += 1

        writer.add_scalar("train/loss_step", loss.item(), global_step)
        if step % args.log_every == 0 or step == len(loader):
            elapsed = time.time() - epoch_start
            print(
                f"epoch={epoch:03d} step={step:04d}/{len(loader):04d} "
                f"loss={loss_sum / sample_count:.4f} "
                f"accuracy={100.0 * correct / sample_count:.2f}% "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    return loss_sum / sample_count, correct / sample_count, global_step


@torch.no_grad()
def evaluate(
    model: VisionTransformer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.shape[0]
        loss_sum += loss.item() * batch_size
        correct += logits.argmax(dim=-1).eq(labels).sum().item()
        sample_count += batch_size

    return loss_sum / sample_count, correct / sample_count


def save_checkpoint(path: Path, state: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA/PPU is not available, but a cuda device was requested.")
    device = torch.device(args.device)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = args.output_dir / "tensorboard"
    writer = SummaryWriter(tensorboard_dir)

    train_loader, validation_loader, test_loader, class_names = build_dataloaders(args, device)
    model, model_config = build_model(args)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_learning_rate,
    )

    start_epoch = 1
    global_step = 0
    best_validation_accuracy = -math.inf
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_validation_accuracy = float(checkpoint.get("best_validation_accuracy", -math.inf))
        print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    run_config = {
        **vars(args),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "resume": None if args.resume is None else str(args.resume),
        "model_config": model_config,
        "parameter_count": parameter_count,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "class_names": class_names,
        "normalization_mean": CIFAR10_MEAN,
        "normalization_std": CIFAR10_STD,
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
    print(f"tensorboard={tensorboard_dir}", flush=True)

    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_accuracy, global_step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                global_step,
                writer,
                args,
            )
            validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
            scheduler.step()

            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("train/accuracy_epoch", train_accuracy, epoch)
            writer.add_scalar("validation/loss", validation_loss, epoch)
            writer.add_scalar("validation/accuracy", validation_accuracy, epoch)
            writer.add_scalar("optimization/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            writer.flush()

            is_best = validation_accuracy > best_validation_accuracy
            if is_best:
                best_validation_accuracy = validation_accuracy

            state = {
                "epoch": epoch,
                "global_step": global_step,
                "best_validation_accuracy": best_validation_accuracy,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "model_config": model_config,
                "class_names": class_names,
                "normalization_mean": CIFAR10_MEAN,
                "normalization_std": CIFAR10_STD,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
            save_checkpoint(latest_path, state)
            if is_best:
                save_checkpoint(best_path, state)

            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} train_acc={100.0 * train_accuracy:.2f}% "
                f"val_loss={validation_loss:.4f} val_acc={100.0 * validation_accuracy:.2f}% "
                f"best_val_acc={100.0 * best_validation_accuracy:.2f}%",
                flush=True,
            )

        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        model.to(device)
        test_loss, test_accuracy = evaluate(model, test_loader, device)
        writer.add_scalar("test/loss", test_loss, 0)
        writer.add_scalar("test/accuracy", test_accuracy, 0)
        writer.flush()
        print(
            f"best_epoch={best_checkpoint['epoch']} "
            f"test_loss={test_loss:.4f} test_acc={100.0 * test_accuracy:.2f}%",
            flush=True,
        )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
