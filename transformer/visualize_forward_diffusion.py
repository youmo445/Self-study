from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_ppu_runtime_path() -> None:
    """Restart once with the shared PPU runtime libraries available."""
    sdk_root = Path("/usr/local/PPU_SDK")
    marker = "_DIFFUSION_VIS_PPU_ENV_READY"
    if not sdk_root.is_dir() or os.environ.get(marker) == "1":
        return

    def prepend_paths(variable: str, new_paths: list[Path]) -> str:
        old_paths = [p for p in os.environ.get(variable, "").split(":") if p]
        paths = [str(p) for p in new_paths if p.is_dir()]
        paths.extend(old_paths)
        return ":".join(dict.fromkeys(paths))

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
    environment[marker] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


if __name__ == "__main__":
    ensure_ppu_runtime_path()


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from diffusion_zsq import (
    make_beta_schedule,
    precompute_diffusion_items,
    q_sample,
)


PROJECT_DIR = Path(__file__).resolve().parent
CIFAR10_CLASSES = (
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
)
TIMESTEPS = torch.tensor([0, 50, 100, 250, 500, 750, 999], dtype=torch.long)


def to_display_image(image: torch.Tensor):
    return (
        ((image.clamp(-1, 1) + 1) / 2)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def main() -> None:
    torch.manual_seed(7)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )
    dataset = datasets.CIFAR10(
        root=PROJECT_DIR / "data" / "cifar10",
        train=False,
        download=False,
        transform=transform,
    )
    image, label = dataset[0]
    x0 = image.unsqueeze(0)

    betas = make_beta_schedule(1000, 1e-4, 2e-2)
    _, _, sqrt_alpha_bars, sqrt_one_minus_alpha_bars = (
        precompute_diffusion_items(betas)
    )

    batch_size = TIMESTEPS.numel()
    x0_batch = x0.repeat(batch_size, 1, 1, 1)
    shared_noise = torch.randn_like(x0).repeat(batch_size, 1, 1, 1)
    xt, _ = q_sample(
        x0_batch,
        TIMESTEPS,
        sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars,
        noise=shared_noise,
    )

    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    axes[0].imshow(to_display_image(x0[0]), interpolation="nearest")
    axes[0].set_title("clean $x_0$")
    axes[0].axis("off")

    for panel, timestep, noisy_image in zip(axes[1:], TIMESTEPS, xt):
        t = int(timestep.item())
        signal = float(sqrt_alpha_bars[t])
        noise = float(sqrt_one_minus_alpha_bars[t])
        panel.imshow(to_display_image(noisy_image), interpolation="nearest")
        panel.set_title(f"t={t}\nsignal={signal:.3f}, noise={noise:.3f}")
        panel.axis("off")

    figure.suptitle(
        f"CIFAR-10 forward diffusion (shared noise) — {CIFAR10_CLASSES[label]}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    output_dir = PROJECT_DIR / "data" / "diffusion_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "forward_noising.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"saved: {output_path}")
    print(f"sample class: {CIFAR10_CLASSES[label]}")
    print(f"timesteps: {TIMESTEPS.tolist()}")


if __name__ == "__main__":
    main()
