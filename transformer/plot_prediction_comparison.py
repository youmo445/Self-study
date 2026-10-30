from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


EPOCH_PATTERN = re.compile(
    r"epoch=(\d+) train_loss=([0-9.eE+-]+) "
    r"val_loss=([0-9.eE+-]+) best_val_loss=([0-9.eE+-]+)"
)
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon-dir", type=Path, required=True)
    parser.add_argument("--v-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_epoch_losses(path: Path) -> tuple[list[int], list[float], list[float]]:
    epochs: list[int] = []
    train_losses: list[float] = []
    validation_losses: list[float] = []
    for match in EPOCH_PATTERN.finditer(path.read_text(encoding="utf-8")):
        epochs.append(int(match.group(1)))
        train_losses.append(float(match.group(2)))
        validation_losses.append(float(match.group(3)))
    return epochs, train_losses, validation_losses


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def plot_curves_and_metrics(
    epsilon_dir: Path,
    v_dir: Path,
    output_path: Path,
) -> None:
    eps_epochs, eps_train, eps_val = read_epoch_losses(epsilon_dir / "train.log")
    v_epochs, v_train, v_val = read_epoch_losses(v_dir / "train.log")
    eps_metrics = load_json(epsilon_dir / "shared_metrics.json")
    v_metrics = load_json(v_dir / "shared_metrics.json")

    eps_bins = eps_metrics["timestep_bins"]
    v_bins = v_metrics["timestep_bins"]
    centers = [(item["t_start"] + item["t_end"]) / 2 for item in eps_bins]

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(eps_epochs, eps_train, label="train", linewidth=1.5)
    axes[0, 0].plot(eps_epochs, eps_val, label="validation", linewidth=1.5)
    axes[0, 0].set_title("Epsilon target loss (not cross-comparable)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(v_epochs, v_train, label="train", linewidth=1.5)
    axes[0, 1].plot(v_epochs, v_val, label="validation", linewidth=1.5)
    axes[0, 1].set_title("v target loss (not cross-comparable)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(
        centers,
        [item["clipped_x0_mse"] for item in eps_bins],
        marker="o",
        label="epsilon prediction",
    )
    axes[1, 0].plot(
        centers,
        [item["clipped_x0_mse"] for item in v_bins],
        marker="o",
        label="v prediction",
    )
    axes[1, 0].set_title("Recovered x0 MSE on identical test inputs")
    axes[1, 0].set_xlabel("Diffusion timestep t")
    axes[1, 0].set_ylabel("Clipped x0 MSE")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(
        centers,
        [item["noise_mse"] for item in eps_bins],
        marker="o",
        label="epsilon prediction",
    )
    axes[1, 1].plot(
        centers,
        [item["noise_mse"] for item in v_bins],
        marker="o",
        label="v prediction",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Recovered noise MSE on identical test inputs")
    axes[1, 1].set_xlabel("Diffusion timestep t")
    axes[1, 1].set_ylabel("Noise MSE (log scale)")
    axes[1, 1].grid(alpha=0.25, which="both")
    axes[1, 1].legend()

    figure.suptitle(
        "Equal-cost epsilon vs v prediction: 200 epochs, seed 42, same DiT",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_samples(epsilon_dir: Path, v_dir: Path, output_path: Path) -> None:
    panels = [
        (
            epsilon_dir / "ddim50_cfg4_seed20261029.png",
            "DDIM-50 · epsilon",
        ),
        (v_dir / "ddim50_cfg4_seed20261029.png", "DDIM-50 · v"),
        (
            epsilon_dir / "ddpm1000_cfg4_seed20261029.png",
            "DDPM-1000 · epsilon",
        ),
        (v_dir / "ddpm1000_cfg4_seed20261029.png", "DDPM-1000 · v"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(9, 16))
    for axis, (path, title) in zip(axes.flat, panels):
        image = plt.imread(path)
        axis.imshow(image, interpolation="nearest")
        axis.set_title(title, fontsize=13)
        axis.set_xticks([])
        height = image.shape[0]
        row_centers = [(index + 0.5) * height / 10 for index in range(10)]
        axis.set_yticks(row_centers)
        axis.set_yticklabels(CLASS_NAMES, fontsize=8)
        axis.tick_params(axis="y", length=0, pad=4)

    figure.suptitle(
        "Matched samples: same seed, CFG=4.0, four images per CIFAR-10 class",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_curves_and_metrics(
        args.epsilon_dir,
        args.v_dir,
        args.output_dir / "curves_and_metrics.png",
    )
    plot_samples(
        args.epsilon_dir,
        args.v_dir,
        args.output_dir / "sample_comparison.png",
    )


if __name__ == "__main__":
    main()
