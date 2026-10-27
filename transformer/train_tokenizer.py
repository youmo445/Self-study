from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "iwslt2017_zh_en"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def normalize_text(text: str) -> str:
    """Collapse newlines and repeated whitespace into single spaces."""
    return " ".join(text.split())


def iter_sentences(
    parquet_path: Path,
    language: str,
    batch_size: int = 4096,
) -> Iterator[str]:
    """Yield one normalized English or Chinese sentence at a time."""
    if language not in {"en", "zh"}:
        raise ValueError(f"language must be 'en' or 'zh', got {language!r}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not parquet_path.is_file():
        raise FileNotFoundError(f"dataset not found: {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["translation"],
    ):
        for row in batch.to_pylist():
            translation = row.get("translation")
            if not translation:
                continue

            text = translation.get(language)
            if not text:
                continue

            text = normalize_text(text)
            if text:
                yield text


def ensure_output_is_available(model_prefix: Path) -> None:
    """Protect an existing tokenizer from accidental overwrite."""
    output_files = [
        model_prefix.with_suffix(".model"),
        model_prefix.with_suffix(".vocab"),
    ]
    existing = [path for path in output_files if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"tokenizer output already exists: {formatted}. "
            "Move or delete it explicitly before retraining."
        )


def train_tokenizer(
    parquet_path: Path,
    language: str,
    model_prefix: Path,
    vocab_size: int = 8000,
    batch_size: int = 4096,
    num_threads: int = 8,
) -> Path:
    """Train one BPE SentencePiece model and return its .model path."""
    if vocab_size <= UNK_ID + 1:
        raise ValueError("vocab_size is too small for the four special tokens")
    if num_threads <= 0:
        raise ValueError("num_threads must be positive")

    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    ensure_output_is_available(model_prefix)

    row_count = pq.ParquetFile(parquet_path).metadata.num_rows
    print(f"\nTraining {language!r} BPE tokenizer")
    print(f"  input rows : {row_count:,}")
    print(f"  vocab size : {vocab_size:,}")
    print(f"  output     : {model_prefix}")

    sentence_iterator = iter_sentences(
        parquet_path=parquet_path,
        language=language,
        batch_size=batch_size,
    )

    spm.SentencePieceTrainer.train(
        sentence_iterator=sentence_iterator,
        model_prefix=str(model_prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=1.0,
        input_sentence_size=0,
        shuffle_input_sentence=False,
        max_sentence_length=4096,
        num_threads=num_threads,
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        unk_id=UNK_ID,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
    )

    return model_prefix.with_suffix(".model")


def inspect_tokenizer(model_path: Path, text: str) -> None:
    """Load a trained tokenizer and print one encode/decode round trip."""
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))

    expected_special_ids = {
        "pad": (tokenizer.pad_id(), PAD_ID),
        "bos": (tokenizer.bos_id(), BOS_ID),
        "eos": (tokenizer.eos_id(), EOS_ID),
        "unk": (tokenizer.unk_id(), UNK_ID),
    }
    for name, (actual, expected) in expected_special_ids.items():
        if actual != expected:
            raise RuntimeError(
                f"{name} id mismatch: expected {expected}, got {actual}"
            )

    core_ids = tokenizer.encode(text, out_type=int)
    token_ids = [tokenizer.bos_id(), *core_ids, tokenizer.eos_id()]
    pieces = [tokenizer.id_to_piece(token_id) for token_id in token_ids]
    decoded_text = tokenizer.decode(core_ids)

    print("\nTokenizer check")
    print(f"  model      : {model_path}")
    print(f"  vocab size : {tokenizer.get_piece_size():,}")
    print(f"  text       : {text}")
    print(f"  pieces     : {pieces}")
    print(f"  ids        : {token_ids}")
    print(f"  decoded    : {decoded_text}")


def check_dataset(train_path: Path, batch_size: int) -> None:
    """Validate the parquet schema and show one sample per language."""
    if not train_path.is_file():
        raise FileNotFoundError(f"dataset not found: {train_path}")

    parquet_file = pq.ParquetFile(train_path)
    names = set(parquet_file.schema_arrow.names)
    if "translation" not in names:
        raise ValueError(
            f"expected a 'translation' column, got {sorted(names)}"
        )

    en_sample = next(iter_sentences(train_path, "en", batch_size))
    zh_sample = next(iter_sentences(train_path, "zh", batch_size))

    print(f"Dataset: {train_path}")
    print(f"Rows   : {parquet_file.metadata.num_rows:,}")
    print(f"EN     : {en_sample}")
    print(f"ZH     : {zh_sample}")


def parse_args() -> argparse.Namespace:
    default_threads = min(8, os.cpu_count() or 1)

    parser = argparse.ArgumentParser(
        description="Train separate English and Chinese BPE tokenizers."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing train.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Tokenizer output directory (default: DATA_DIR/tokenizers)",
    )
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-threads", type=int, default=default_threads)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and preview the dataset without training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    train_path = data_dir / "train.parquet"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else data_dir / "tokenizers"
    )

    check_dataset(train_path, args.batch_size)
    if args.check_only:
        print("\nDataset check passed; tokenizer training was not started.")
        return

    en_model_path = train_tokenizer(
        parquet_path=train_path,
        language="en",
        model_prefix=output_dir / "spm_en",
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )
    inspect_tokenizer(
        en_model_path,
        "I really enjoy learning about transformers.",
    )

    zh_model_path = train_tokenizer(
        parquet_path=train_path,
        language="zh",
        model_prefix=output_dir / "spm_zh",
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )
    inspect_tokenizer(
        zh_model_path,
        "我非常喜欢学习Transformer。",
    )

    print("\nBoth tokenizers were trained successfully.")
    print(f"English model: {en_model_path}")
    print(f"Chinese model: {zh_model_path}")


if __name__ == "__main__":
    main()
