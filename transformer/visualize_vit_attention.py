from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def ensure_ppu_runtime_path() -> None:
    """Restart once with the PPU SDK runtime configured before importing torch."""
    sdk_root = Path("/usr/local/PPU_SDK")
    marker = "_VIT_ATTENTION_PPU_ENV_READY"
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
    )
    environment["PYTHONUNBUFFERED"] = "1"
    environment[marker] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


if __name__ == "__main__":
    ensure_ppu_runtime_path()


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from vit_zsq import VisionTransformer


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "cifar10"
DEFAULT_CHECKPOINT = PROJECT_DIR / "data" / "checkpoints" / "vit_cifar10" / "best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "vit_attention_outputs"
DEFAULT_CLASS_NAMES = [
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
    parser = argparse.ArgumentParser(
        description="Predict CIFAR-10 images and visualize CLS-to-patch attention."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Encoder layer index. Negative indices count from the end; -1 is the last layer.",
    )
    parser.add_argument(
        "--head",
        default="mean",
        help="Attention head index, or 'mean' to average all heads.",
    )
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


def select_head(cls_to_patch: torch.Tensor, head: str) -> tuple[torch.Tensor, str]:
    # cls_to_patch: [B, num_heads, num_patches]
    if head == "mean":
        return cls_to_patch.mean(dim=1), "mean_heads"

    head_index = int(head)
    if not 0 <= head_index < cls_to_patch.shape[1]:
        raise ValueError(
            f"head={head_index} is out of range for {cls_to_patch.shape[1]} attention heads"
        )
    return cls_to_patch[:, head_index, :], f"head_{head_index}"


def denormalize(image: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=image.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype).view(3, 1, 1)
    return (image.cpu() * std_tensor + mean_tensor).clamp(0.0, 1.0)


def save_visualization(
    output_path: Path,
    image: torch.Tensor,
    raw_patch_attention: torch.Tensor,
    true_name: str,
    predicted_name: str,
    predicted_probability: float,
    layer_index: int,
    head_description: str,
) -> None:
    image_array = image.permute(1, 2, 0).numpy()
    raw_grid = raw_patch_attention.numpy()

    minimum = raw_patch_attention.min()
    maximum = raw_patch_attention.max()
    display_grid = (raw_patch_attention - minimum) / (maximum - minimum + 1e-12)
    smooth_map = F.interpolate(
        display_grid[None, None],
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_array)
    axes[0].set_title(f"true: {true_name}\npred: {predicted_name} ({predicted_probability:.3f})")
    axes[0].axis("off")

    heatmap_image = axes[1].imshow(raw_grid, cmap="magma", interpolation="nearest")
    axes[1].set_title(f"raw CLS -> patch\nlayer {layer_index}, {head_description}")
    axes[1].set_xlabel("patch column")
    axes[1].set_ylabel("patch row")
    figure.colorbar(heatmap_image, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(image_array)
    axes[2].imshow(smooth_map, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
    axes[2].set_title("relative attention overlay")
    axes[2].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA/PPU is not available, but a cuda device was requested.")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    model = VisionTransformer(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    mean = tuple(checkpoint.get("normalization_mean", (0.4914, 0.4822, 0.4465)))
    std = tuple(checkpoint.get("normalization_std", (0.2470, 0.2435, 0.2616)))
    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=transform,
        download=False,
    )

    for index in args.indices:
        if not 0 <= index < len(test_dataset):
            raise IndexError(f"test index {index} is outside [0, {len(test_dataset) - 1}]")

    layer_count = len(model.encoder.layers)
    layer_index = args.layer if args.layer >= 0 else layer_count + args.layer
    if not 0 <= layer_index < layer_count:
        raise ValueError(f"layer={args.layer} resolves to {layer_index}, but model has {layer_count} layers")

    normalized_images = []
    labels = []
    display_images = []
    for index in args.indices:
        image, label = test_dataset[index]
        normalized_images.append(image)
        labels.append(label)
        display_images.append(denormalize(image, mean, std))
    image_batch = torch.stack(normalized_images).to(device)

    captured: dict[str, torch.Tensor] = {}

    def capture_attention(_module, _inputs, output) -> None:
        # MultiHeadAttention returns (context, weights).
        captured["weights"] = output[1].detach().cpu()

    handle = model.encoder.layers[layer_index].attn.register_forward_hook(capture_attention)
    try:
        with torch.no_grad():
            logits = model(image_batch)
            probabilities = logits.softmax(dim=-1).cpu()
    finally:
        handle.remove()

    if "weights" not in captured:
        raise RuntimeError("attention hook did not capture weights")

    # [B, heads, query_tokens, key_tokens] -> CLS query row -> patch key columns.
    attention_weights = captured["weights"]
    cls_self_attention = attention_weights[:, :, 0, 0]
    cls_to_patch_all_heads = attention_weights[:, :, 0, 1:]
    selected_attention, head_description = select_head(cls_to_patch_all_heads, args.head)

    patch_size = int(model_config["patch_size"])
    patch_rows = int(model_config["H"]) // patch_size
    patch_columns = int(model_config["W"]) // patch_size
    expected_patches = patch_rows * patch_columns
    if selected_attention.shape[-1] != expected_patches:
        raise RuntimeError(
            f"captured {selected_attention.shape[-1]} patches, expected {expected_patches}"
        )
    selected_attention = selected_attention.reshape(-1, patch_rows, patch_columns)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    topk = min(args.topk, len(class_names))
    top_probabilities, top_indices = probabilities.topk(topk, dim=-1)
    summaries = []

    for batch_index, dataset_index in enumerate(args.indices):
        true_label = labels[batch_index]
        predicted_label = int(probabilities[batch_index].argmax().item())
        predicted_probability = float(probabilities[batch_index, predicted_label].item())
        raw_grid = selected_attention[batch_index]

        image_path = args.output_dir / f"test_{dataset_index:05d}_layer_{layer_index}_{head_description}.png"
        tensor_path = args.output_dir / f"test_{dataset_index:05d}_attention.pt"
        save_visualization(
            image_path,
            display_images[batch_index],
            raw_grid,
            class_names[true_label],
            class_names[predicted_label],
            predicted_probability,
            layer_index,
            head_description,
        )
        torch.save(
            {
                "test_index": dataset_index,
                "layer": layer_index,
                "head": args.head,
                "all_heads_cls_to_patch": cls_to_patch_all_heads[batch_index],
                "selected_patch_grid": raw_grid,
                "cls_self_attention_per_head": cls_self_attention[batch_index],
            },
            tensor_path,
        )

        top_predictions = [
            {
                "class": class_names[int(class_index)],
                "probability": float(probability),
            }
            for probability, class_index in zip(
                top_probabilities[batch_index].tolist(),
                top_indices[batch_index].tolist(),
            )
        ]
        summary = {
            "test_index": dataset_index,
            "true_class": class_names[true_label],
            "predicted_class": class_names[predicted_label],
            "predicted_probability": predicted_probability,
            "top_predictions": top_predictions,
            "patch_attention_sum": float(raw_grid.sum().item()),
            "image_path": str(image_path),
            "raw_attention_path": str(tensor_path),
        }
        summaries.append(summary)

        print(
            f"index={dataset_index} true={class_names[true_label]} "
            f"pred={class_names[predicted_label]} probability={predicted_probability:.4f}"
        )
        print("CLS -> patch attention grid:")
        print(np.array2string(raw_grid.numpy(), precision=5, suppress_small=False))
        print(f"saved: {image_path}")

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)
    print(f"summary: {summary_path}")
    print(
        "Note: the overlay is min-max normalized only for display; the .pt files preserve raw softmax scores."
    )


if __name__ == "__main__":
    main()
