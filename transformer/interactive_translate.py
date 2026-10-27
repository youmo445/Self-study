from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path


def ensure_ppu_environment() -> None:
    """Restart once with the server's official PPU environment."""
    sdk_root = Path("/usr/local/PPU_SDK")
    marker = "_INTERACTIVE_TRANSLATE_PPU_ENV_READY"
    if not sdk_root.is_dir() or os.environ.get(marker) == "1":
        return

    def prepend_paths(variable: str, new_paths: list[Path]) -> str:
        current_paths = [
            path
            for path in os.environ.get(variable, "").split(":")
            if path
        ]
        combined = [str(path) for path in new_paths if path.is_dir()]
        combined.extend(current_paths)
        return ":".join(dict.fromkeys(combined))

    environment = os.environ.copy()
    environment["PPU_SDK"] = str(sdk_root)
    environment["PPU_PATH"] = str(sdk_root)
    environment["PPU_HOME"] = str(sdk_root)
    environment["SAILSHMEM_DIR"] = str(sdk_root / "sailSHMEM")
    environment["PATH"] = prepend_paths(
        "PATH",
        [
            sdk_root / "bin",
            sdk_root / "asight" / "bin",
            sdk_root / "ppu-smi" / "bin",
        ],
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
    os.execve(
        sys.executable,
        [sys.executable, *sys.argv],
        environment,
    )


if __name__ == "__main__":
    ensure_ppu_environment()


import torch

from train_translation import (
    ModelConfig,
    load_checkpoint,
    load_tokenizer,
    translate_texts,
)
from transformer_zsq import Transformer


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "iwslt2017_zh_en"
DEFAULT_CHECKPOINT = (
    DEFAULT_DATA_DIR / "checkpoints" / "iwslt_en_zh" / "best.pt"
)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Transformer, ModelConfig, dict]:
    checkpoint = load_checkpoint(checkpoint_path)
    model_config = ModelConfig(**checkpoint["model_config"])

    model = Transformer(**asdict(model_config))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, model_config, checkpoint


def run_interactive(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint.resolve()
    data_dir = args.data_dir.resolve()
    device = torch.device(args.device)

    src_tokenizer = load_tokenizer(
        data_dir / "tokenizers" / "spm_en.model"
    )
    tgt_tokenizer = load_tokenizer(
        data_dir / "tokenizers" / "spm_zh.model"
    )
    model, model_config, checkpoint = load_model(checkpoint_path, device)

    if src_tokenizer.get_piece_size() != model_config.src_vocab_size:
        raise ValueError("source tokenizer does not match best.pt")
    if tgt_tokenizer.get_piece_size() != model_config.tgt_vocab_size:
        raise ValueError("target tokenizer does not match best.pt")

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = model_config.max_len - 1
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    epoch = checkpoint.get("epoch", "unknown")
    best_val_loss = checkpoint.get("best_val_loss")
    print(f"Loaded checkpoint : {checkpoint_path}")
    print(f"Device            : {device}")
    print(f"Checkpoint epoch  : {epoch}")
    if best_val_loss is not None:
        print(f"Best val loss     : {float(best_val_loss):.4f}")
    print("\n输入英文并回车；输入 q 退出。")

    try:
        while True:
            english = input("\nEN> ").strip()
            if english.lower() in {"q", "quit", "exit"}:
                print("已退出。")
                break
            if not english:
                continue

            prediction = translate_texts(
                model=model,
                texts=[english],
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                max_len=model_config.max_len,
                max_new_tokens=max_new_tokens,
                device=device,
            )[0]
            print(f"ZH> {prediction}")
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive English-to-Chinese translation."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_interactive(args)


if __name__ == "__main__":
    main()
