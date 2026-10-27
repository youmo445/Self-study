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
    """Restart once with the settings from PPU_SDK/envsetup.sh ppu."""
    sdk_root = Path("/usr/local/PPU_SDK")
    marker = "_TRANSFORMER_PPU_ENV_READY"
    if not sdk_root.is_dir() or os.environ.get(marker) == "1":
        return

    def prepend_paths(variable: str, new_paths: list[Path]) -> str:
        old_paths = [
            path
            for path in os.environ.get(variable, "").split(":")
            if path
        ]
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
    os.execve(
        sys.executable,
        [sys.executable, *sys.argv],
        environment,
    )


# This must happen before importing torch, whose PPU backend loads
# libhggcrt1.so during import.
if __name__ == "__main__":
    ensure_ppu_runtime_path()


import pyarrow.parquet as pq
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from transformer_zsq import Transformer, make_padding_mask, shift_targets


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "iwslt2017_zh_en"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def normalize_text(text: str) -> str:
    """Use the same whitespace normalization as tokenizer training."""
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
    expected_ids = {
        "pad": PAD_ID,
        "bos": BOS_ID,
        "eos": EOS_ID,
        "unk": UNK_ID,
    }
    if actual_ids != expected_ids:
        raise ValueError(
            f"special-token ids in {model_path} are {actual_ids}, "
            f"expected {expected_ids}"
        )
    return tokenizer


class TranslationDataset(Dataset):
    """IWSLT English-to-Chinese examples, encoded one row at a time."""

    def __init__(
        self,
        parquet_path: Path,
        src_tokenizer_path: Path,
        tgt_tokenizer_path: Path,
        max_len: int,
        max_samples: int | None = None,
    ) -> None:
        if not parquet_path.is_file():
            raise FileNotFoundError(f"dataset not found: {parquet_path}")
        if max_len < 3:
            raise ValueError("max_len must be at least 3 (BOS + content + EOS)")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive")

        table = pq.read_table(parquet_path, columns=["translation"])
        self.translations = table["translation"]
        self.max_len = max_len
        self.src_tokenizer = load_tokenizer(src_tokenizer_path)
        self.tgt_tokenizer = load_tokenizer(tgt_tokenizer_path)
        self._length = len(self.translations)
        if max_samples is not None:
            self._length = min(self._length, max_samples)

    def __len__(self) -> int:
        return self._length

    def _encode(
        self,
        text: str,
        tokenizer: spm.SentencePieceProcessor,
    ) -> Tensor:
        core_ids = tokenizer.encode(normalize_text(text), out_type=int)
        core_ids = core_ids[: self.max_len - 2]
        token_ids = [tokenizer.bos_id(), *core_ids, tokenizer.eos_id()]
        return torch.tensor(token_ids, dtype=torch.long)

    def raw_pair(self, index: int) -> tuple[str, str]:
        translation = self.translations[index].as_py()
        src_text = translation.get("en")
        tgt_text = translation.get("zh")
        if not src_text or not tgt_text:
            raise ValueError(f"empty translation at row {index}: {translation}")
        return normalize_text(src_text), normalize_text(tgt_text)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        src_text, tgt_text = self.raw_pair(index)
        return {
            "src_token_ids": self._encode(src_text, self.src_tokenizer),
            "tgt_token_ids": self._encode(tgt_text, self.tgt_tokenizer),
        }


@dataclass(frozen=True)
class TranslationCollator:
    src_pad_id: int
    tgt_pad_id: int

    def __call__(self, examples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        if not examples:
            raise ValueError("cannot collate an empty batch")

        src_token_ids = pad_sequence(
            [example["src_token_ids"] for example in examples],
            batch_first=True,
            padding_value=self.src_pad_id,
        )
        tgt_token_ids = pad_sequence(
            [example["tgt_token_ids"] for example in examples],
            batch_first=True,
            padding_value=self.tgt_pad_id,
        )
        return {
            "src_token_ids": src_token_ids,
            "tgt_token_ids": tgt_token_ids,
        }


@dataclass(frozen=True)
class ModelConfig:
    src_pad_id: int
    tgt_pad_id: int
    src_vocab_size: int
    tgt_vocab_size: int
    max_len: int
    d_model: int
    d_ff: int
    num_head: int
    num_layers: int
    dropout: float


def validate_model_config(config: ModelConfig) -> None:
    if config.d_model <= 0 or config.d_ff <= 0:
        raise ValueError("d_model and d_ff must be positive")
    if config.num_head <= 0 or config.num_layers <= 0:
        raise ValueError("num_head and num_layers must be positive")
    if config.d_model % config.num_head != 0:
        raise ValueError("d_model must be divisible by num_head")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")


def initialize_transformer(model: Transformer, d_model: int) -> None:
    """Initialization suitable for embeddings followed by sqrt(d_model)."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=d_model**-0.5)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def noam_learning_rate(
    step: int,
    d_model: int,
    warmup_steps: int,
    scale: float,
) -> float:
    if step <= 0:
        raise ValueError("optimizer step must be positive")
    return scale * (d_model**-0.5) * min(
        step**-0.5,
        step * warmup_steps**-1.5,
    )


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def batch_loss(
    model: Transformer,
    src_token_ids: Tensor,
    tgt_token_ids: Tensor,
    tgt_pad_id: int,
) -> tuple[Tensor, Tensor, int]:
    decoder_input, labels = shift_targets(tgt_token_ids)
    logits, _, _, _ = model(src_token_ids, decoder_input)

    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=tgt_pad_id,
        reduction="sum",
    )
    token_count = int(labels.ne(tgt_pad_id).sum().item())
    if token_count == 0:
        raise RuntimeError("batch contains no non-padding target tokens")

    mean_loss = loss_sum / token_count
    return mean_loss, loss_sum.detach(), token_count


def train_one_epoch(
    model: Transformer,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tgt_pad_id: int,
    d_model: int,
    warmup_steps: int,
    learning_rate_scale: float,
    grad_clip: float,
    global_step: int,
    log_every: int,
    max_steps: int | None,
    start_batch: int,
    writer: SummaryWriter,
) -> tuple[float, int, int]:
    model.train()
    total_loss_sum = 0.0
    total_tokens = 0
    interval_loss_sum = 0.0
    interval_tokens = 0
    interval_start = time.perf_counter()
    processed_batches = 0
    last_batch_index = start_batch

    for batch_index, batch in enumerate(data_loader, start=1):
        if batch_index <= start_batch:
            continue
        if max_steps is not None and processed_batches >= max_steps:
            break
        processed_batches += 1
        last_batch_index = batch_index

        src_token_ids = batch["src_token_ids"].to(device)
        tgt_token_ids = batch["tgt_token_ids"].to(device)

        global_step += 1
        learning_rate = noam_learning_rate(
            step=global_step,
            d_model=d_model,
            warmup_steps=warmup_steps,
            scale=learning_rate_scale,
        )
        set_optimizer_learning_rate(optimizer, learning_rate)

        optimizer.zero_grad(set_to_none=True)
        loss, loss_sum, token_count = batch_loss(
            model=model,
            src_token_ids=src_token_ids,
            tgt_token_ids=tgt_token_ids,
            tgt_pad_id=tgt_pad_id,
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"non-finite loss at global step {global_step}: {loss.item()}"
            )

        loss.backward()
        gradient_norm: float | None = None
        if grad_clip > 0.0:
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )
            gradient_norm = float(gradient_norm_tensor.item())
        optimizer.step()

        numeric_loss_sum = float(loss_sum.item())
        step_loss = numeric_loss_sum / token_count
        total_loss_sum += numeric_loss_sum
        total_tokens += token_count
        interval_loss_sum += numeric_loss_sum
        interval_tokens += token_count

        writer.add_scalar("Loss/train_step", step_loss, global_step)
        writer.add_scalar(
            "Perplexity/train_step",
            math.exp(min(step_loss, 20.0)),
            global_step,
        )
        writer.add_scalar(
            "Optimization/learning_rate",
            learning_rate,
            global_step,
        )
        writer.add_scalar("Batch/target_tokens", token_count, global_step)
        if gradient_norm is not None:
            writer.add_scalar(
                "Optimization/gradient_norm_before_clip",
                gradient_norm,
                global_step,
            )

        if global_step % log_every == 0:
            elapsed = max(time.perf_counter() - interval_start, 1e-9)
            interval_loss = interval_loss_sum / interval_tokens
            perplexity = math.exp(min(interval_loss, 20.0))
            tokens_per_second = interval_tokens / elapsed
            print(
                f"  step={global_step:,} "
                f"batch={batch_index:,}/{len(data_loader):,} "
                f"loss={interval_loss:.4f} "
                f"ppl={perplexity:.2f} "
                f"lr={learning_rate:.3e} "
                f"tokens/s={tokens_per_second:,.0f}",
                flush=True,
            )
            writer.add_scalar(
                "Speed/target_tokens_per_second",
                tokens_per_second,
                global_step,
            )
            interval_loss_sum = 0.0
            interval_tokens = 0
            interval_start = time.perf_counter()

    if total_tokens == 0:
        raise RuntimeError("training epoch processed no target tokens")
    return total_loss_sum / total_tokens, global_step, last_batch_index


@torch.no_grad()
def evaluate(
    model: Transformer,
    data_loader: DataLoader,
    device: torch.device,
    tgt_pad_id: int,
    max_steps: int | None = None,
) -> float:
    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0

    for batch_index, batch in enumerate(data_loader, start=1):
        if max_steps is not None and batch_index > max_steps:
            break

        src_token_ids = batch["src_token_ids"].to(device)
        tgt_token_ids = batch["tgt_token_ids"].to(device)
        _, loss_sum, token_count = batch_loss(
            model=model,
            src_token_ids=src_token_ids,
            tgt_token_ids=tgt_token_ids,
            tgt_pad_id=tgt_pad_id,
        )
        total_loss_sum += float(loss_sum.item())
        total_tokens += token_count

    if total_tokens == 0:
        raise RuntimeError("validation processed no target tokens")
    return total_loss_sum / total_tokens


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src_token_ids: Tensor,
    src_pad_id: int,
    tgt_bos_id: int,
    tgt_eos_id: int,
    max_new_tokens: int,
) -> Tensor:
    model.eval()
    src_mask = make_padding_mask(src_token_ids, src_pad_id)
    memory, _ = model.encoder(src_token_ids, src_mask)

    batch_size = src_token_ids.shape[0]
    generated = torch.full(
        (batch_size, 1),
        tgt_bos_id,
        dtype=torch.long,
        device=src_token_ids.device,
    )
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=src_token_ids.device,
    )

    for _ in range(max_new_tokens):
        decoder_output, _, _ = model.decoder(
            generated,
            memory,
            src_mask=src_mask,
        )
        next_logits = model.out_proj(decoder_output[:, -1, :])
        next_token = next_logits.argmax(dim=-1)
        next_token = torch.where(
            finished,
            torch.full_like(next_token, tgt_eos_id),
            next_token,
        )
        generated = torch.cat([generated, next_token[:, None]], dim=1)
        finished = finished | next_token.eq(tgt_eos_id)
        if bool(finished.all().item()):
            break

    return generated


def encode_source_texts(
    texts: list[str],
    tokenizer: spm.SentencePieceProcessor,
    max_len: int,
) -> Tensor:
    sequences = []
    for text in texts:
        core_ids = tokenizer.encode(normalize_text(text), out_type=int)
        core_ids = core_ids[: max_len - 2]
        sequences.append(
            torch.tensor(
                [tokenizer.bos_id(), *core_ids, tokenizer.eos_id()],
                dtype=torch.long,
            )
        )
    return pad_sequence(
        sequences,
        batch_first=True,
        padding_value=tokenizer.pad_id(),
    )


def decode_target_ids(
    token_ids: list[int],
    tokenizer: spm.SentencePieceProcessor,
) -> str:
    content_ids = []
    for token_id in token_ids:
        if token_id == tokenizer.bos_id():
            continue
        if token_id == tokenizer.eos_id():
            break
        if token_id != tokenizer.pad_id():
            content_ids.append(token_id)
    return tokenizer.decode(content_ids)


def translate_texts(
    model: Transformer,
    texts: list[str],
    src_tokenizer: spm.SentencePieceProcessor,
    tgt_tokenizer: spm.SentencePieceProcessor,
    max_len: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    if not texts:
        return []
    max_new_tokens = min(max_new_tokens, max_len - 1)
    src_token_ids = encode_source_texts(
        texts=texts,
        tokenizer=src_tokenizer,
        max_len=max_len,
    ).to(device)
    generated = greedy_decode(
        model=model,
        src_token_ids=src_token_ids,
        src_pad_id=src_tokenizer.pad_id(),
        tgt_bos_id=tgt_tokenizer.bos_id(),
        tgt_eos_id=tgt_tokenizer.eos_id(),
        max_new_tokens=max_new_tokens,
    )
    return [
        decode_target_ids(row.tolist(), tgt_tokenizer)
        for row in generated.cpu()
    ]


def show_validation_translations(
    model: Transformer,
    dataset: TranslationDataset,
    sample_count: int,
    max_new_tokens: int,
    device: torch.device,
) -> None:
    if sample_count <= 0:
        return
    sample_count = min(sample_count, len(dataset))
    if sample_count == 1:
        indices = [0]
    else:
        indices = [
            round(i * (len(dataset) - 1) / (sample_count - 1))
            for i in range(sample_count)
        ]

    pairs = [dataset.raw_pair(index) for index in indices]
    sources = [source for source, _ in pairs]
    predictions = translate_texts(
        model=model,
        texts=sources,
        src_tokenizer=dataset.src_tokenizer,
        tgt_tokenizer=dataset.tgt_tokenizer,
        max_len=dataset.max_len,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    print("\nValidation translations")
    for index, ((source, reference), prediction) in enumerate(
        zip(pairs, predictions),
        start=1,
    ):
        print(f"  [{index}] EN  : {source}")
        print(f"      REF : {reference}")
        print(f"      PRED: {prediction}")


def checkpoint_payload(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    include_optimizer: bool,
    batch_in_epoch: int,
    epoch_complete: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 2,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "epoch_complete": epoch_complete,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_config": asdict(model_config),
        "model_state_dict": model.state_dict(),
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def make_data_loader(
    dataset: TranslationDataset,
    collator: TranslationCollator,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    options: dict[str, Any] = {}
    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        generator=generator,
        **options,
    )


def run_train(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if args.warmup_steps <= 0 or args.log_every <= 0:
        raise ValueError("warmup_steps and log_every must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    set_seed(args.seed)
    device = torch.device(args.device)
    data_dir = args.data_dir.resolve()
    src_tokenizer_path = data_dir / "tokenizers" / "spm_en.model"
    tgt_tokenizer_path = data_dir / "tokenizers" / "spm_zh.model"

    train_dataset = TranslationDataset(
        parquet_path=data_dir / "train.parquet",
        src_tokenizer_path=src_tokenizer_path,
        tgt_tokenizer_path=tgt_tokenizer_path,
        max_len=args.max_len,
        max_samples=args.max_train_samples,
    )
    validation_dataset = TranslationDataset(
        parquet_path=data_dir / "validation.parquet",
        src_tokenizer_path=src_tokenizer_path,
        tgt_tokenizer_path=tgt_tokenizer_path,
        max_len=args.max_len,
        max_samples=args.max_val_samples,
    )

    model_config = ModelConfig(
        src_pad_id=train_dataset.src_tokenizer.pad_id(),
        tgt_pad_id=train_dataset.tgt_tokenizer.pad_id(),
        src_vocab_size=train_dataset.src_tokenizer.get_piece_size(),
        tgt_vocab_size=train_dataset.tgt_tokenizer.get_piece_size(),
        max_len=args.max_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_head=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    validate_model_config(model_config)

    collator = TranslationCollator(
        src_pad_id=model_config.src_pad_id,
        tgt_pad_id=model_config.tgt_pad_id,
    )
    validation_loader = make_data_loader(
        dataset=validation_dataset,
        collator=collator,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = Transformer(**asdict(model_config))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    start_epoch = 1
    resume_batch_in_epoch = 0
    global_step = 0
    best_val_loss = math.inf
    if args.resume is None:
        initialize_transformer(model, model_config.d_model)
    else:
        checkpoint = load_checkpoint(args.resume.resolve())
        saved_config = ModelConfig(**checkpoint["model_config"])
        if saved_config != model_config:
            raise ValueError(
                "resume checkpoint model configuration does not match the "
                f"current arguments\ncheckpoint: {saved_config}\ncurrent: {model_config}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Restored model and optimizer state.")
        else:
            print("Restored model state; checkpoint has no optimizer state.")
        saved_epoch = int(checkpoint["epoch"])
        if bool(checkpoint.get("epoch_complete", True)):
            start_epoch = saved_epoch + 1
        else:
            start_epoch = saved_epoch
            resume_batch_in_epoch = int(
                checkpoint.get("batch_in_epoch", 0)
            )
            print(
                f"Resuming epoch {start_epoch} after batch "
                f"{resume_batch_in_epoch:,}."
            )
        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))

    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    output_dir = args.output_dir.resolve()
    checkpoint_path = output_dir / "best.pt"
    latest_checkpoint_path = output_dir / "latest.pt"
    tensorboard_dir = (
        args.tensorboard_dir.resolve()
        if args.tensorboard_dir is not None
        else output_dir / "tensorboard"
    )

    print("Training configuration")
    print(f"  device       : {device}")
    print(f"  train samples: {len(train_dataset):,}")
    print(f"  valid samples: {len(validation_dataset):,}")
    print(f"  vocabulary   : {model_config.src_vocab_size:,} -> {model_config.tgt_vocab_size:,}")
    print(f"  max length   : {model_config.max_len}")
    print(f"  batch size   : {args.batch_size}")
    print(f"  model        : d={model_config.d_model}, ff={model_config.d_ff}, heads={model_config.num_head}, layers={model_config.num_layers}")
    print(f"  parameters   : {parameter_count:,}")
    print(f"  best model   : {checkpoint_path}")
    print(f"  latest state : {latest_checkpoint_path}")
    print("  save interval: at the end of every epoch")
    print(f"  TensorBoard  : {tensorboard_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model_config": asdict(model_config),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "warmup_steps": args.warmup_steps,
        "learning_rate_scale": args.learning_rate_scale,
        "tensorboard_dir": str(tensorboard_dir),
        "seed": args.seed,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    writer_options: dict[str, Any] = {
        "log_dir": str(tensorboard_dir),
        "max_queue": 10,
        "flush_secs": 10,
    }
    if args.resume is not None:
        writer_options["purge_step"] = global_step + 1
    writer = SummaryWriter(**writer_options)
    writer.add_text(
        "Run/configuration",
        "```json\n"
        + json.dumps(run_config, indent=2, ensure_ascii=False)
        + "\n```",
        global_step,
    )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loader = make_data_loader(
                dataset=train_dataset,
                collator=collator,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                seed=args.seed + epoch - 1,
            )
            start_batch = (
                resume_batch_in_epoch if epoch == start_epoch else 0
            )

            epoch_start = time.perf_counter()
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_loss, global_step, last_batch_index = train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                tgt_pad_id=model_config.tgt_pad_id,
                d_model=model_config.d_model,
                warmup_steps=args.warmup_steps,
                learning_rate_scale=args.learning_rate_scale,
                grad_clip=args.grad_clip,
                global_step=global_step,
                log_every=args.log_every,
                max_steps=args.max_train_steps_per_epoch,
                start_batch=start_batch,
                writer=writer,
            )
            resume_batch_in_epoch = 0
            val_loss = evaluate(
                model=model,
                data_loader=validation_loader,
                device=device,
                tgt_pad_id=model_config.tgt_pad_id,
                max_steps=args.max_val_steps,
            )
            val_perplexity = math.exp(min(val_loss, 20.0))
            elapsed = time.perf_counter() - epoch_start
            print(
                f"Epoch {epoch} complete: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_ppl={val_perplexity:.2f}, "
                f"time={elapsed / 60:.1f} min"
            )

            writer.add_scalar("Loss/train_epoch", train_loss, global_step)
            writer.add_scalar("Loss/validation", val_loss, global_step)
            writer.add_scalar(
                "Perplexity/validation",
                val_perplexity,
                global_step,
            )
            writer.add_scalar("Progress/epoch", epoch, global_step)
            writer.add_scalar(
                "Time/epoch_minutes",
                elapsed / 60.0,
                global_step,
            )

            show_validation_translations(
                model=model,
                dataset=validation_dataset,
                sample_count=args.sample_count,
                max_new_tokens=args.sample_max_new_tokens,
                device=device,
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            if not args.no_save:
                if is_best:
                    best_payload = checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        model_config=model_config,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        include_optimizer=args.save_optimizer,
                        batch_in_epoch=last_batch_index,
                        epoch_complete=True,
                    )
                    save_checkpoint(checkpoint_path, best_payload)
                    print(
                        f"Saved new best checkpoint: {checkpoint_path}"
                    )

                latest_payload = checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    model_config=model_config,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    include_optimizer=True,
                    batch_in_epoch=last_batch_index,
                    epoch_complete=True,
                )
                save_checkpoint(latest_checkpoint_path, latest_payload)
                print(
                    f"Saved end-of-epoch state: {latest_checkpoint_path}"
                )
            writer.flush()
    finally:
        writer.close()

    print("\nTraining finished.")
    if not args.no_save:
        print(f"Best checkpoint: {checkpoint_path}")
        print(f"Latest resumable checkpoint: {latest_checkpoint_path}")
        print(
            "Translate with:\n"
            f"  python {Path(__file__).name} translate "
            f"--checkpoint {checkpoint_path} "
            '--text "I love learning about artificial intelligence."'
        )


def run_translate(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint.resolve())
    model_config = ModelConfig(**checkpoint["model_config"])
    validate_model_config(model_config)

    data_dir = args.data_dir.resolve()
    src_tokenizer = load_tokenizer(
        data_dir / "tokenizers" / "spm_en.model"
    )
    tgt_tokenizer = load_tokenizer(
        data_dir / "tokenizers" / "spm_zh.model"
    )
    if src_tokenizer.get_piece_size() != model_config.src_vocab_size:
        raise ValueError("source tokenizer vocabulary does not match checkpoint")
    if tgt_tokenizer.get_piece_size() != model_config.tgt_vocab_size:
        raise ValueError("target tokenizer vocabulary does not match checkpoint")

    device = torch.device(args.device)
    model = Transformer(**asdict(model_config))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = model_config.max_len - 1
    predictions = translate_texts(
        model=model,
        texts=args.text,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_len=model_config.max_len,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    for source, prediction in zip(args.text, predictions):
        print(f"EN: {source}")
        print(f"ZH: {prediction}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run the hand-written Transformer on IWSLT2017 en-zh."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train the model")
    train_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / "checkpoints" / "iwslt_en_zh",
    )
    train_parser.add_argument("--device", default="cuda:1")
    train_parser.add_argument("--epochs", type=int, default=1000)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--max-len", type=int, default=96)
    train_parser.add_argument("--d-model", type=int, default=256)
    train_parser.add_argument("--d-ff", type=int, default=1024)
    train_parser.add_argument("--num-heads", type=int, default=8)
    train_parser.add_argument("--num-layers", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--warmup-steps", type=int, default=4000)
    train_parser.add_argument("--learning-rate-scale", type=float, default=1.0)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--log-every", type=int, default=100)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--sample-count", type=int, default=2)
    train_parser.add_argument("--sample-max-new-tokens", type=int, default=64)
    train_parser.add_argument("--resume", type=Path, default=None)
    train_parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="event directory (default: OUTPUT_DIR/tensorboard)",
    )
    train_parser.add_argument("--save-optimizer", action="store_true")
    train_parser.add_argument("--no-save", action="store_true")
    train_parser.add_argument("--max-train-samples", type=int, default=None)
    train_parser.add_argument("--max-val-samples", type=int, default=None)
    train_parser.add_argument(
        "--max-train-steps-per-epoch",
        type=int,
        default=None,
    )
    train_parser.add_argument("--max-val-steps", type=int, default=None)
    train_parser.set_defaults(handler=run_train)

    translate_parser = subparsers.add_parser(
        "translate",
        help="translate English text with a checkpoint",
    )
    translate_parser.add_argument("--checkpoint", type=Path, required=True)
    translate_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    translate_parser.add_argument("--device", default="cuda:1")
    translate_parser.add_argument(
        "--text",
        action="append",
        required=True,
        help="English text; repeat this option for multiple sentences",
    )
    translate_parser.add_argument("--max-new-tokens", type=int, default=None)
    translate_parser.set_defaults(handler=run_translate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
