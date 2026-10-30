from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path


def ensure_ppu_runtime_path() -> None:
    sdk_root = Path("/usr/local/PPU_SDK")
    marker = "_FLOW_DIT_SAMPLE_PPU_ENV_READY"
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
from torchvision.utils import save_image

from dit_zsq import DiT
from flow_matching_zsq import flow_euler_sample


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "data" / "checkpoints" / "dit_compare_flow_seed42" / "best.pt"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "checkpoints" / "dit_compare_flow_seed42" / "samples" / "euler_20.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a Flow Matching DiT with Euler integration.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--num-sample-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261029)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if args.num_sample_steps <= 0:
        raise ValueError("--num-sample-steps must be positive.")
    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("training_objective") != "flow_matching":
        raise ValueError("The checkpoint is not a Flow Matching checkpoint.")

    model_config = checkpoint["model_config"]
    class_names = checkpoint.get("class_names", [str(index) for index in range(10)])
    model = DiT(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    num_classes = int(model_config["num_classes"])
    labels = torch.arange(num_classes, dtype=torch.long).repeat_interleave(
        args.samples_per_class
    )
    sample_shape = (
        labels.shape[0],
        int(model_config["out_channels"]),
        int(model_config["image_height"]),
        int(model_config["image_width"]),
    )
    print(
        f"checkpoint={args.checkpoint} epoch={checkpoint['epoch']} device={device} "
        f"samples={labels.shape[0]} solver=euler steps={args.num_sample_steps} "
        f"guidance_scale={args.guidance_scale}",
        flush=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    sample_start = time.perf_counter()
    samples = flow_euler_sample(
        model,
        sample_shape,
        labels,
        args.num_sample_steps,
        args.guidance_scale,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    sample_seconds = time.perf_counter() - sample_start

    samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(samples.cpu(), args.output, nrow=args.samples_per_class)
    print(f"saved={args.output}", flush=True)
    print(
        f"sampling_seconds={sample_seconds:.3f} "
        f"seconds_per_step={sample_seconds / args.num_sample_steps:.4f}",
        flush=True,
    )
    print("row_labels=" + ", ".join(class_names[:num_classes]), flush=True)


if __name__ == "__main__":
    main()
