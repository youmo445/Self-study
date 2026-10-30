from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


EPOCH_PATTERN = re.compile(
    r"epoch=(\d+) train_loss=([0-9.eE+-]+) "
    r"val_loss=([0-9.eE+-]+) best_val_loss=([0-9.eE+-]+)"
)
BEST_PATTERN = re.compile(r"best_epoch=(\d+) test_loss=([0-9.eE+-]+)")
TIME_PATTERN = re.compile(r"sampling_seconds=([0-9.]+)")
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
    text = path.read_text(encoding="utf-8")
    matches = list(EPOCH_PATTERN.finditer(text))
    return (
        [int(match.group(1)) for match in matches],
        [float(match.group(2)) for match in matches],
        [float(match.group(3)) for match in matches],
    )


def read_best_and_test(path: Path) -> tuple[int, float]:
    matches = list(BEST_PATTERN.finditer(path.read_text(encoding="utf-8")))
    if not matches:
        raise ValueError(f"No final test line in {path}")
    return int(matches[-1].group(1)), float(matches[-1].group(2))


def read_seconds(path: Path) -> float:
    match = TIME_PATTERN.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"No sampling_seconds in {path}")
    return float(match.group(1))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def image_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def add_class_ticks(axis: plt.Axes, image: np.ndarray) -> None:
    height = image.shape[0]
    centers = [(index + 0.5) * height / 10 for index in range(10)]
    axis.set_yticks(centers)
    axis.set_yticklabels(CLASS_NAMES, fontsize=6)
    axis.tick_params(axis="y", length=0, pad=2)
    axis.set_xticks([])


def plot_sample_comparison(results_dir: Path) -> None:
    entries = [
        ("DDIM-50 · epsilon", "epsilon_ddim50_cfg4.png"),
        ("DDIM-50 · x0", "x0_ddim50_cfg4.png"),
        ("DDIM-50 · v", "v_ddim50_cfg4.png"),
        ("Flow Euler-50", "euler50_cfg4_seed20261029.png"),
        ("Flow Euler-5", "euler5_cfg4_seed20261029.png"),
        ("Flow Euler-10", "euler10_cfg4_seed20261029.png"),
        ("Flow Euler-20", "euler20_cfg4_seed20261029.png"),
        ("Flow Euler-50", "euler50_cfg4_seed20261029.png"),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 15.5))
    for axis, (title, filename) in zip(axes.flat, entries):
        image = plt.imread(results_dir / filename)
        axis.imshow(image, interpolation="nearest")
        axis.set_title(title, fontsize=11)
        add_class_ticks(axis, image)
    axes[0, 0].set_ylabel("Matched 50-step comparison", fontsize=10, labelpad=35)
    axes[1, 0].set_ylabel("Flow step ablation", fontsize=10, labelpad=35)
    figure.suptitle(
        "Same seed=20261029, CFG=4.0, four samples per CIFAR-10 class",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        results_dir / "flow_vs_diffusion_samples.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def plot_metrics(results_dir: Path, flow_metrics: dict, timings: dict[str, float]) -> dict:
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10))

    epochs, train_loss, validation_loss = read_epoch_losses(results_dir / "flow_train.log")
    axes[0, 0].plot(epochs, train_loss, label="train velocity MSE", linewidth=1.5)
    axes[0, 0].plot(epochs, validation_loss, label="validation velocity MSE", linewidth=1.5)
    axes[0, 0].set_title("Flow Matching training curve")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    bins = flow_metrics["time_bins"]
    centers = [(item["t_start"] + item["t_end"]) / 2 for item in bins]
    axes[0, 1].plot(
        centers,
        [item["velocity_mse"] for item in bins],
        marker="o",
        label="velocity MSE",
    )
    axes[0, 1].plot(
        centers,
        [item["x0_mse_on_flow_path"] for item in bins],
        marker="o",
        label="recovered x0 MSE",
    )
    axes[0, 1].plot(
        centers,
        [item["noise_mse_on_flow_path"] for item in bins],
        marker="o",
        label="recovered noise MSE",
    )
    axes[0, 1].set_title("Flow-path test error by continuous time")
    axes[0, 1].set_xlabel("Flow time t (noise 0 → data 1)")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    time_labels = ["eps\nDDIM-50", "x0\nDDIM-50", "v\nDDIM-50", "Flow-5", "Flow-10", "Flow-20", "Flow-50"]
    time_keys = ["epsilon_50", "x0_50", "v_50", "flow_5", "flow_10", "flow_20", "flow_50"]
    time_values = [timings[key] for key in time_keys]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"] + ["#9467bd"] * 4
    bars = axes[1, 0].bar(time_labels, time_values, color=colors)
    axes[1, 0].bar_label(bars, fmt="%.3fs", padding=3, fontsize=8)
    axes[1, 0].set_ylim(0, max(time_values) * 1.25)
    axes[1, 0].set_title("End-to-end cold-process sampling time (40 images)")
    axes[1, 0].set_ylabel("Seconds (lower is better)")
    axes[1, 0].grid(axis="y", alpha=0.25)

    flow50 = image_array(results_dir / "euler50_cfg4_seed20261029.png")
    flow_steps = [5, 10, 20, 50]
    rendered_mse = []
    for step in flow_steps:
        current = image_array(results_dir / f"euler{step}_cfg4_seed20261029.png")
        rendered_mse.append(float(np.mean((current - flow50) ** 2)))
    axes[1, 1].plot(flow_steps, rendered_mse, marker="o", linewidth=2, color="#9467bd")
    axes[1, 1].set_yscale("symlog", linthresh=1e-7)
    axes[1, 1].set_title("Flow Euler convergence to its 50-step output")
    axes[1, 1].set_xlabel("Euler steps (CFG uses 2 model forwards per step)")
    axes[1, 1].set_ylabel("Rendered grid MSE vs Flow-50")
    axes[1, 1].set_xticks(flow_steps)
    axes[1, 1].grid(alpha=0.25, which="both")

    figure.suptitle(
        "Equal training cost: 200 epochs / 70,400 optimizer steps / same 7.41M DiT",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        results_dir / "flow_vs_diffusion_metrics.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)
    return {str(step): mse for step, mse in zip(flow_steps, rendered_mse)}


def write_summary(
    results_dir: Path,
    flow_metrics: dict,
    timings: dict[str, float],
    rendered_mse: dict[str, float],
) -> None:
    old_metrics = {
        name: load_json(results_dir / f"{name}_metrics.json")
        for name in ("epsilon", "x0", "v")
    }
    models = {}
    for name in ("epsilon", "x0", "v", "flow"):
        best_epoch, test_loss = read_best_and_test(results_dir / f"{name}_train.log")
        models[name] = {
            "best_epoch": best_epoch,
            "test_target_mse": test_loss,
            "target_mse_warning": "Only comparable within the same training target/path.",
        }
    for name, metric in old_metrics.items():
        models[name]["diffusion_path_metrics"] = metric["overall"]
    models["flow"]["flow_path_metrics"] = flow_metrics["overall"]

    summary = {
        "training_cost": {
            "epochs": 200,
            "optimizer_steps": 70400,
            "model_parameters": 7408176,
            "train_validation_test_samples": [45000, 5000, 10000],
            "seed": 42,
        },
        "sampling_setup": {
            "seed": 20261029,
            "guidance_scale": 4.0,
            "samples_per_class": 4,
            "num_classes": 10,
            "cfg_model_forwards_per_solver_step": 2,
        },
        "models": models,
        "sampling_seconds": timings,
        "flow_rendered_grid_mse_vs_flow50": rendered_mse,
        "fairness_note": (
            "Diffusion-path and Flow-path reconstruction MSE are not direct cross-method "
            "scores because their x_t distributions differ. Compare equal training cost, "
            "matched generated grids, solver steps/model forwards, and sampling time."
        ),
    }
    with (results_dir / "four_way_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    flow_metrics = load_json(args.results_dir / "flow_metrics.json")
    timings = {
        "epsilon_50": 2.522,
        "x0_50": 2.541,
        "v_50": 2.509,
        "flow_5": read_seconds(args.results_dir / "sample_euler5.log"),
        "flow_10": read_seconds(args.results_dir / "sample_euler10.log"),
        "flow_20": read_seconds(args.results_dir / "sample_euler20.log"),
        "flow_50": read_seconds(args.results_dir / "sample_euler50.log"),
    }
    plot_sample_comparison(args.results_dir)
    rendered_mse = plot_metrics(args.results_dir, flow_metrics, timings)
    write_summary(args.results_dir, flow_metrics, timings, rendered_mse)


if __name__ == "__main__":
    main()
