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
PREDICTION_TYPES = ("epsilon", "x0", "v")
DISPLAY_NAMES = {
    "epsilon": "epsilon-prediction",
    "x0": "x0-prediction",
    "v": "v-prediction",
}
COLORS = {"epsilon": "#1f77b4", "x0": "#2ca02c", "v": "#ff7f0e"}
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
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
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


def plot_curves_and_metrics(results_dir: Path, metrics: dict[str, dict]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.5))

    for column, prediction_type in enumerate(PREDICTION_TYPES):
        epochs, train_loss, validation_loss = read_epoch_losses(
            results_dir / f"{prediction_type}_train.log"
        )
        axis = axes[0, column]
        axis.plot(epochs, train_loss, label="train", linewidth=1.4)
        axis.plot(epochs, validation_loss, label="validation", linewidth=1.4)
        axis.set_title(f"{DISPLAY_NAMES[prediction_type]} target loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE (not cross-comparable)")
        axis.grid(alpha=0.25)
        axis.legend()

    reference_bins = metrics["epsilon"]["timestep_bins"]
    centers = [
        (item["t_start"] + item["t_end"]) / 2 for item in reference_bins
    ]

    for prediction_type in PREDICTION_TYPES:
        bins = metrics[prediction_type]["timestep_bins"]
        axes[1, 0].plot(
            centers,
            [item["clipped_x0_mse"] for item in bins],
            marker="o",
            linewidth=1.8,
            color=COLORS[prediction_type],
            label=DISPLAY_NAMES[prediction_type],
        )
        axes[1, 1].plot(
            centers,
            [item["noise_mse"] for item in bins],
            marker="o",
            linewidth=1.8,
            color=COLORS[prediction_type],
            label=DISPLAY_NAMES[prediction_type],
        )

    axes[1, 0].set_title("Recovered x0 on identical test inputs")
    axes[1, 0].set_xlabel("Diffusion timestep t")
    axes[1, 0].set_ylabel("Clipped x0 MSE")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Recovered noise on identical test inputs")
    axes[1, 1].set_xlabel("Diffusion timestep t")
    axes[1, 1].set_ylabel("Noise MSE (log scale)")
    axes[1, 1].grid(alpha=0.25, which="both")
    axes[1, 1].legend(fontsize=8)

    psnr_values = [
        metrics[prediction_type]["overall"]["clipped_x0_psnr"]
        for prediction_type in PREDICTION_TYPES
    ]
    bars = axes[1, 2].bar(
        [DISPLAY_NAMES[item] for item in PREDICTION_TYPES],
        psnr_values,
        color=[COLORS[item] for item in PREDICTION_TYPES],
    )
    axes[1, 2].set_title("Overall clipped x0 PSNR")
    axes[1, 2].set_ylabel("PSNR (dB, higher is better)")
    axes[1, 2].set_ylim(min(psnr_values) - 0.25, max(psnr_values) + 0.25)
    axes[1, 2].tick_params(axis="x", rotation=15)
    axes[1, 2].grid(axis="y", alpha=0.25)
    axes[1, 2].bar_label(bars, fmt="%.3f dB", padding=3, fontsize=9)

    figure.suptitle(
        "Equal-cost epsilon vs x0 vs v: 200 epochs, seed 42, same 7.41M DiT",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        results_dir / "three_way_curves_and_metrics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_samples(results_dir: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 15.5))
    for row, sampler in enumerate(("ddim50", "ddpm1000")):
        for column, prediction_type in enumerate(PREDICTION_TYPES):
            axis = axes[row, column]
            path = results_dir / f"{prediction_type}_{sampler}_cfg4.png"
            image = plt.imread(path)
            axis.imshow(image, interpolation="nearest")
            sampler_name = "DDIM-50" if sampler == "ddim50" else "DDPM-1000"
            axis.set_title(
                f"{sampler_name} · {DISPLAY_NAMES[prediction_type]}",
                fontsize=12,
            )
            axis.set_xticks([])
            height = image.shape[0]
            row_centers = [(index + 0.5) * height / 10 for index in range(10)]
            axis.set_yticks(row_centers)
            axis.set_yticklabels(CLASS_NAMES, fontsize=7)
            axis.tick_params(axis="y", length=0, pad=3)

    figure.suptitle(
        "Matched samples: same seed, CFG=4.0, four images per CIFAR-10 class",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        results_dir / "three_way_sample_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_summary(results_dir: Path, metrics: dict[str, dict]) -> None:
    summary = {
        prediction_type: {
            "checkpoint_epoch": metrics[prediction_type]["checkpoint_epoch"],
            **metrics[prediction_type]["overall"],
        }
        for prediction_type in PREDICTION_TYPES
    }
    with (results_dir / "three_way_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    metrics = {
        prediction_type: load_json(args.results_dir / f"{prediction_type}_metrics.json")
        for prediction_type in PREDICTION_TYPES
    }
    plot_curves_and_metrics(args.results_dir, metrics)
    plot_samples(args.results_dir)
    write_summary(args.results_dir, metrics)


if __name__ == "__main__":
    main()
