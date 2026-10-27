from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def ensure_ppu_runtime_path() -> None:
    """Restart once after adding the PPU SDK paths required by torch."""
    sdk_root = Path("/usr/local/PPU_SDK")
    marker = "_FNN_PPU_ENV_READY"
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


import pyarrow.parquet as pq
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "iwslt2017_zh_en"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
REFERENCE_TRANSFORMER_PARAMETERS = 13_524_800


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def load_tokenizer(model_path: Path) -> spm.SentencePieceProcessor:
    if not model_path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {model_path}")
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
    actual_ids = {
        "pad": tokenizer.pad_id(),
        "bos": tokenizer.bos_id(),
        "eos": tokenizer.eos_id(),
        "unk": tokenizer.unk_id(),
    }
    expected_ids = {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID, "unk": UNK_ID}
    if actual_ids != expected_ids:
        raise ValueError(f"special token ids are {actual_ids}, expected {expected_ids}")
    return tokenizer


class ChineseSentenceDataset(Dataset[Tensor]):
    """Tokenized Chinese sentences stored as one flat tensor plus offsets."""

    def __init__(
        self,
        parquet_path: Path,
        tokenizer_path: Path,
        context_size: int,
        max_sentence_tokens: int,
        cache_dir: Path,
        max_samples: int | None = None,
    ) -> None:
        if not parquet_path.is_file():
            raise FileNotFoundError(f"dataset not found: {parquet_path}")
        if context_size <= 0 or max_sentence_tokens <= 0:
            raise ValueError("context_size and max_sentence_tokens must be positive")

        tokenizer = load_tokenizer(tokenizer_path)
        self.context_size = context_size
        cache_path = cache_dir / (
            f"{parquet_path.stem}.zh.context_{context_size}.max_{max_sentence_tokens}.pt"
        )

        if max_samples is None and cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu")
            self.token_ids = cached["token_ids"]
            self.offsets = cached["offsets"]
            print(
                f"Loaded token cache {cache_path} "
                f"({len(self):,} sentences, {self.token_ids.numel():,} stored ids)",
                flush=True,
            )
            return

        table = pq.read_table(parquet_path, columns=["translation"])
        translations = table["translation"]
        sample_count = len(translations)
        if max_samples is not None:
            sample_count = min(sample_count, max_samples)

        flat_ids: list[int] = []
        offsets = [0]
        prefix = [BOS_ID] * context_size
        started_at = time.perf_counter()
        for index in range(sample_count):
            translation = translations[index].as_py()
            chinese = translation.get("zh")
            if chinese is None:
                raise ValueError(f"row {index} has no Chinese text")
            content_ids = tokenizer.encode(normalize_text(chinese), out_type=int)
            content_ids = content_ids[:max_sentence_tokens]
            sentence_ids = [*prefix, *content_ids, EOS_ID]
            flat_ids.extend(sentence_ids)
            offsets.append(len(flat_ids))

        self.token_ids = torch.tensor(flat_ids, dtype=torch.long)
        self.offsets = torch.tensor(offsets, dtype=torch.long)
        elapsed = time.perf_counter() - started_at
        print(
            f"Tokenized {len(self):,} sentences from {parquet_path.name} "
            f"in {elapsed:.1f}s ({self.token_ids.numel():,} stored ids)",
            flush=True,
        )

        if max_samples is None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            torch.save(
                {"token_ids": self.token_ids, "offsets": self.offsets},
                temporary_path,
            )
            os.replace(temporary_path, cache_path)
            print(f"Saved token cache to {cache_path}", flush=True)

    def __len__(self) -> int:
        return int(self.offsets.numel() - 1)

    def __getitem__(self, index: int) -> Tensor:
        start = int(self.offsets[index].item())
        end = int(self.offsets[index + 1].item())
        return self.token_ids[start:end]


@dataclass(frozen=True)
class NGramCollator:
    context_size: int

    def __call__(self, sentences: list[Tensor]) -> tuple[Tensor, Tensor]:
        if not sentences:
            raise ValueError("cannot collate an empty batch")
        windows = [
            sentence.unfold(0, self.context_size + 1, 1)
            for sentence in sentences
        ]
        all_windows = torch.cat(windows, dim=0)
        contexts = all_windows[:, :-1].contiguous()
        targets = all_windows[:, -1].contiguous()
        return contexts, targets


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_size: int
    embedding_dim: int
    hidden_dim: int


class FeedForwardLanguageModel(torch.nn.Module):
    """Bengio-style fixed-context neural language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=PAD_ID,
        )
        self.hidden = torch.nn.Linear(
            config.context_size * config.embedding_dim,
            config.hidden_dim,
        )
        self.output = torch.nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, context_ids: Tensor) -> Tensor:
        # [N, C] -> [N, C, E] -> [N, C*E]
        embedded = self.embedding(context_ids)
        flattened = embedded.reshape(embedded.shape[0], -1)
        hidden_state = torch.tanh(self.hidden(flattened))
        # These are raw scores. Cross entropy performs log-softmax internally.
        return self.output(hidden_state)


def initialize_model(model: FeedForwardLanguageModel) -> None:
    torch.nn.init.normal_(model.embedding.weight, mean=0.0, std=0.02)
    with torch.no_grad():
        model.embedding.weight[PAD_ID].zero_()
    torch.nn.init.xavier_uniform_(model.hidden.weight)
    torch.nn.init.zeros_(model.hidden.bias)
    torch.nn.init.xavier_uniform_(model.output.weight)
    torch.nn.init.zeros_(model.output.bias)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def train_one_epoch(
    model: FeedForwardLanguageModel,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    grad_clip: float,
    log_every: int,
    writer: SummaryWriter,
) -> tuple[float, int, int]:
    model.train()
    total_loss_sum = 0.0
    total_targets = 0
    interval_loss_sum = 0.0
    interval_targets = 0
    interval_started_at = time.perf_counter()

    for batch_index, (contexts, targets) in enumerate(data_loader, start=1):
        contexts = contexts.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(contexts)
        loss_sum = F.cross_entropy(logits, targets, reduction="sum")
        target_count = targets.numel()
        loss = loss_sum / target_count
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite loss at step {global_step + 1}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        global_step += 1
        numeric_loss_sum = float(loss_sum.detach().item())
        step_loss = numeric_loss_sum / target_count
        total_loss_sum += numeric_loss_sum
        total_targets += target_count
        interval_loss_sum += numeric_loss_sum
        interval_targets += target_count

        writer.add_scalar("Loss/train_step", step_loss, global_step)
        writer.add_scalar("Perplexity/train_step", perplexity(step_loss), global_step)
        writer.add_scalar("Batch/predicted_tokens", target_count, global_step)
        writer.add_scalar(
            "Optimization/gradient_norm_before_clip",
            float(gradient_norm.item()),
            global_step,
        )

        if global_step % log_every == 0:
            elapsed = max(time.perf_counter() - interval_started_at, 1e-9)
            interval_loss = interval_loss_sum / interval_targets
            tokens_per_second = interval_targets / elapsed
            print(
                f"epoch={epoch:02d} step={global_step:,} "
                f"batch={batch_index:,}/{len(data_loader):,} "
                f"loss={interval_loss:.4f} ppl={perplexity(interval_loss):.2f} "
                f"tokens/s={tokens_per_second:,.0f}",
                flush=True,
            )
            writer.add_scalar("Speed/predicted_tokens_per_second", tokens_per_second, global_step)
            interval_loss_sum = 0.0
            interval_targets = 0
            interval_started_at = time.perf_counter()

    return total_loss_sum / total_targets, total_targets, global_step


@torch.no_grad()
def evaluate(
    model: FeedForwardLanguageModel,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[float, int]:
    model.eval()
    total_loss_sum = 0.0
    total_targets = 0
    for contexts, targets in data_loader:
        contexts = contexts.to(device)
        targets = targets.to(device)
        loss_sum = F.cross_entropy(model(contexts), targets, reduction="sum")
        total_loss_sum += float(loss_sum.item())
        total_targets += targets.numel()
    return total_loss_sum / total_targets, total_targets


@torch.no_grad()
def generate_sample(
    model: FeedForwardLanguageModel,
    tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    max_new_tokens: int = 40,
) -> str:
    model.eval()
    generated = [BOS_ID] * model.config.context_size
    for _ in range(max_new_tokens):
        context = torch.tensor(
            [generated[-model.config.context_size :]],
            dtype=torch.long,
            device=device,
        )
        next_id = int(model(context).argmax(dim=-1).item())
        if next_id == EOS_ID:
            break
        generated.append(next_id)
    content_ids = [token_id for token_id in generated[model.config.context_size :] if token_id >= 4]
    return tokenizer.decode(content_ids)


def checkpoint_payload(
    model: FeedForwardLanguageModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    validation_loss: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "validation_loss": validation_loss,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_args": vars(args),
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fixed-context Chinese feed-forward LM")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1280)
    parser.add_argument("--max-sentence-tokens", type=int, default=95)
    parser.add_argument("--batch-size", type=int, default=256, help="sentences per batch")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = args.data_dir / "checkpoints" / "fnn_zh_lm"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = args.output_dir / "tensorboard"
    cache_dir = args.data_dir / "cache"

    set_seed(args.seed)
    tokenizer_path = args.data_dir / "tokenizers" / "spm_zh.model"
    tokenizer = load_tokenizer(tokenizer_path)
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size(),
        context_size=args.context_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    model = FeedForwardLanguageModel(model_config)
    initialize_model(model)
    model_parameters = parameter_count(model)
    difference = model_parameters - REFERENCE_TRANSFORMER_PARAMETERS
    difference_percent = 100.0 * difference / REFERENCE_TRANSFORMER_PARAMETERS
    print(f"FNN trainable parameters:         {model_parameters:,}", flush=True)
    print(f"Transformer trainable parameters: {REFERENCE_TRANSFORMER_PARAMETERS:,}", flush=True)
    print(f"Difference: {difference:+,} ({difference_percent:+.3f}%)", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("the torch CUDA/PPU backend is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print(f"Training device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    model.to(device)

    train_dataset = ChineseSentenceDataset(
        args.data_dir / "train.parquet",
        tokenizer_path,
        args.context_size,
        args.max_sentence_tokens,
        cache_dir,
        args.max_train_samples,
    )
    validation_dataset = ChineseSentenceDataset(
        args.data_dir / "validation.parquet",
        tokenizer_path,
        args.context_size,
        args.max_sentence_tokens,
        cache_dir,
        args.max_validation_samples,
    )
    collator = NGramCollator(args.context_size)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 1
    global_step = 0
    best_validation_loss = math.inf
    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"cannot resume; checkpoint not found: {latest_path}")
        checkpoint = torch.load(latest_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        print(f"Resumed after epoch {start_epoch - 1} at step {global_step:,}", flush=True)

    run_config = {
        "model_config": asdict(model_config),
        "trainable_parameters": model_parameters,
        "reference_transformer_parameters": REFERENCE_TRANSFORMER_PARAMETERS,
        "parameter_difference": difference,
        "parameter_difference_percent": difference_percent,
        "train_sentences": len(train_dataset),
        "validation_sentences": len(validation_dataset),
        "batch_size_in_sentences": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "seed": args.seed,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    writer = SummaryWriter(log_dir=str(tensorboard_dir), purge_step=global_step or None)
    print(f"TensorBoard directory: {tensorboard_dir}", flush=True)
    print(f"Checkpoints: {args.output_dir}", flush=True)
    training_started_at = time.perf_counter()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_started_at = time.perf_counter()
            train_loss, train_targets, global_step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                global_step,
                args.grad_clip,
                args.log_every,
                writer,
            )
            validation_loss, validation_targets = evaluate(model, validation_loader, device)
            epoch_seconds = time.perf_counter() - epoch_started_at

            is_best = validation_loss < best_validation_loss
            if is_best:
                best_validation_loss = validation_loss
            payload = checkpoint_payload(
                model,
                optimizer,
                epoch,
                global_step,
                best_validation_loss,
                validation_loss,
                args,
            )
            atomic_torch_save(payload, latest_path)
            if is_best:
                atomic_torch_save(payload, best_path)

            writer.add_scalar("Loss/train_epoch", train_loss, epoch)
            writer.add_scalar("Loss/validation_epoch", validation_loss, epoch)
            writer.add_scalar("Perplexity/train_epoch", perplexity(train_loss), epoch)
            writer.add_scalar("Perplexity/validation_epoch", perplexity(validation_loss), epoch)
            writer.add_scalar("Epoch/seconds", epoch_seconds, epoch)
            writer.flush()

            sample = generate_sample(model, tokenizer, device)
            print(
                f"epoch={epoch:02d}/{args.epochs} complete in {epoch_seconds:.1f}s | "
                f"train_loss={train_loss:.4f} train_ppl={perplexity(train_loss):.2f} "
                f"({train_targets:,} targets) | val_loss={validation_loss:.4f} "
                f"val_ppl={perplexity(validation_loss):.2f} ({validation_targets:,} targets) | "
                f"best={'yes' if is_best else 'no'}",
                flush=True,
            )
            print(f"greedy sample: {sample!r}", flush=True)
    finally:
        writer.close()

    total_seconds = time.perf_counter() - training_started_at
    print(f"Training finished in {total_seconds / 3600.0:.2f} hours", flush=True)
    print(f"Best checkpoint: {best_path}", flush=True)


if __name__ == "__main__":
    main()
