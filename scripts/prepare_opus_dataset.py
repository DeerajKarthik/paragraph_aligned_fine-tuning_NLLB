#!/usr/bin/env python3
"""Build a reproducible, filtered German-to-English OPUS Books dataset.

The script downloads the ``de-en`` configuration from
``Helsinki-NLP/opus_books`` with the Hugging Face ``datasets`` library.  It
keeps the JSON-array schema already consumed by ``finetune_nllb.py`` and also
writes JSONL, metadata, and MD5 manifests for reproducibility.

Install the required packages before running:

    pip install datasets langdetect rapidfuzz transformers torch

Example:

    python scripts/prepare_opus_dataset.py --output-dir data

The optional language-model score is computed on the English target with
``distilgpt2``.  Use ``--quality-model none`` for a quicker development run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from datasets import load_dataset
from rapidfuzz.fuzz import ratio as levenshtein_ratio

try:
    from langdetect import DetectorFactory, detect_langs
except ImportError:  # Kept optional so --help and smoke tests remain usable.
    DetectorFactory = None
    detect_langs = None


LOGGER = logging.getLogger(__name__)
if DetectorFactory is not None:
    DetectorFactory.seed = 0  # langdetect otherwise changes its result between runs.

SOURCE_LANG = "de"
TARGET_LANG = "en"
SOURCE_NLLB_LANG = "deu_Latn"
TARGET_NLLB_LANG = "eng_Latn"
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MARKUP_RE = re.compile(r"<[^>]+>|&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class Pair:
    """A normalized translation pair and its quality score."""

    source: str
    target: str
    quality: float = 0.0

    @property
    def average_length(self) -> float:
        return (len(self.source) + len(self.target)) / 2

    def as_record(self) -> dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "source_lang": SOURCE_LANG,
            "target_lang": TARGET_LANG,
            "src_lang_nllb": SOURCE_NLLB_LANG,
            "tgt_lang_nllb": TARGET_NLLB_LANG,
        }


def normalize_text(value: Any) -> str | None:
    """Validate, Unicode-normalize, and whitespace-normalize one segment."""
    if not isinstance(value, str):
        return None
    try:
        value.encode("utf-8", "strict")
    except UnicodeError:
        return None
    value = unicodedata.normalize("NFC", value)
    if CONTROL_RE.search(value):
        return None
    value = " ".join(value.split())
    return value or None


def normalized_for_comparison(value: str) -> str:
    """Normalize harmless formatting differences for duplicate detection."""
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(PUNCTUATION_RE.sub(" ", value).split())


def has_expected_language(text: str, expected: str, confidence: float) -> bool:
    """Return false only for a confident, contradictory language prediction."""
    if detect_langs is None:
        raise RuntimeError(
            "Language identification requires langdetect. Install it with "
            "`pip install langdetect`, or use --skip-language-id for a smoke test."
        )
    try:
        predictions = detect_langs(text)
    except Exception:  # Very short or unusual strings are left to other filters.
        return True
    if not predictions:
        return True
    best = predictions[0]
    return not (best.lang != expected and best.probability >= confidence)


def is_clean_pair(pair: Pair, args: argparse.Namespace, counters: Counter[str]) -> bool:
    """Perform UTF-8, length, ratio, and language-ID validation."""
    source_length, target_length = len(pair.source), len(pair.target)
    if not (args.min_chars <= source_length <= args.max_chars):
        counters["source_length"] += 1
        return False
    if not (args.min_chars <= target_length <= args.max_chars):
        counters["target_length"] += 1
        return False
    length_ratio = target_length / source_length
    if not args.min_length_ratio <= length_ratio <= args.max_length_ratio:
        counters["length_ratio"] += 1
        return False
    if not args.skip_language_id:
        if not has_expected_language(pair.source, SOURCE_LANG, args.language_confidence):
            counters["source_language"] += 1
            return False
        if not has_expected_language(pair.target, TARGET_LANG, args.language_confidence):
            counters["target_language"] += 1
            return False
    return True


def load_raw_pairs(dataset_name: str, config: str, split: str, limit: int | None) -> Iterable[Pair]:
    """Load the Hugging Face translation mapping without relying on file layout."""
    dataset = load_dataset(dataset_name, config, split=split)
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        translation = row.get("translation", {})
        source = normalize_text(translation.get(SOURCE_LANG))
        target = normalize_text(translation.get(TARGET_LANG))
        if source is None or target is None:
            continue
        yield Pair(source=source, target=target)


def deduplicate(pairs: Sequence[Pair], threshold: float, counters: Counter[str]) -> list[Pair]:
    """Remove exact and high-similarity duplicates without quadratic matching.

    Exact duplicates are identified after case, punctuation, and whitespace
    normalization.  Fuzzy candidates are blocked by normalized prefixes or
    suffixes and compatible lengths; only those candidates are evaluated with
    normalized Levenshtein similarity.
    """
    unique: list[Pair] = []
    normalized_pairs: list[tuple[str, str]] = []
    exact_seen: set[tuple[str, str]] = set()
    blocks: defaultdict[tuple[str, str], list[int]] = defaultdict(list)

    for pair in pairs:
        source = normalized_for_comparison(pair.source)
        target = normalized_for_comparison(pair.target)
        key = (source, target)
        if key in exact_seen:
            counters["exact_duplicate"] += 1
            continue

        # A >= 0.95 match cannot differ greatly in length.  Prefix/suffix
        # blocking catches near-identical sentences while keeping this stage
        # practical for a corpus with tens of thousands of pairs.
        candidate_ids: set[int] = set()
        for value in (source, target):
            if value:
                candidate_ids.update(blocks[("prefix", value[:12])])
                candidate_ids.update(blocks[("suffix", value[-12:])])
        duplicate = False
        for candidate_id in candidate_ids:
            old_source, old_target = normalized_pairs[candidate_id]
            source_ratio = min(len(source), len(old_source)) / max(len(source), len(old_source), 1)
            target_ratio = min(len(target), len(old_target)) / max(len(target), len(old_target), 1)
            if source_ratio < threshold or target_ratio < threshold:
                continue
            if (
                levenshtein_ratio(source, old_source) / 100 >= threshold
                and levenshtein_ratio(target, old_target) / 100 >= threshold
            ):
                duplicate = True
                break
        if duplicate:
            counters["fuzzy_duplicate"] += 1
            continue

        new_id = len(unique)
        unique.append(pair)
        normalized_pairs.append(key)
        exact_seen.add(key)
        for value in (source, target):
            if value:
                blocks[("prefix", value[:12])].append(new_id)
                blocks[("suffix", value[-12:])].append(new_id)
    return unique


def artifact_cleanliness(pair: Pair) -> float:
    """Penalize markup and control-like artifacts remaining after validation."""
    text = f"{pair.source} {pair.target}"
    artifacts = len(MARKUP_RE.findall(text)) + text.count("�")
    return 1.0 / (1.0 + artifacts)


def untranslated_penalty(pair: Pair) -> float:
    """A high character overlap is suspicious for a German-to-English pair."""
    overlap = levenshtein_ratio(
        normalized_for_comparison(pair.source), normalized_for_comparison(pair.target)
    ) / 100
    return 1.0 - overlap


def target_perplexities(texts: Sequence[str], model_name: str, batch_size: int) -> list[float]:
    """Calculate target-language perplexity with a causal language model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Scoring English targets with %s on %s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    perplexities: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        mask = attention_mask[:, 1:]
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
        ).reshape(labels.shape)
        mean_losses = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        perplexities.extend(torch.exp(mean_losses.clamp(max=20)).cpu().tolist())
    return perplexities


def score_pairs(pairs: Sequence[Pair], model_name: str | None, batch_size: int) -> list[Pair]:
    """Combine untranslated-content, artifact, and target-PPL quality signals."""
    if model_name:
        perplexities = target_perplexities([pair.target for pair in pairs], model_name, batch_size)
        # Lower perplexity is better; log scaling avoids one outlier dominating.
        language_scores = [1 / (1 + math.log1p(value)) for value in perplexities]
    else:
        LOGGER.warning("Skipping language-model scoring because --quality-model is none")
        language_scores = [0.5] * len(pairs)

    scored: list[Pair] = []
    for pair, language_score in zip(pairs, language_scores):
        quality = (
            0.40 * untranslated_penalty(pair)
            + 0.25 * artifact_cleanliness(pair)
            + 0.35 * language_score
        )
        scored.append(Pair(source=pair.source, target=pair.target, quality=quality))
    return scored


def keep_top_quality(pairs: Sequence[Pair], discard_fraction: float) -> list[Pair]:
    """Discard the requested lowest-quality fraction deterministically."""
    if not pairs:
        return []
    discard_count = int(len(pairs) * discard_fraction)
    ranked = sorted(pairs, key=lambda pair: (pair.quality, pair.source, pair.target))
    return ranked[discard_count:]


def stratified_sample(pairs: Sequence[Pair], sample_size: int, seed: int, strata: int = 20) -> list[Pair]:
    """Take an even sample from length strata, preserving the length profile."""
    if len(pairs) < sample_size:
        raise ValueError(
            f"Only {len(pairs):,} pairs remain after filtering; need {sample_size:,}. "
            "Relax a filter or request a smaller --target-size."
        )
    ordered = sorted(pairs, key=lambda pair: (pair.average_length, pair.source, pair.target))
    rng = random.Random(seed)
    selected: list[Pair] = []
    for stratum in range(min(strata, len(ordered))):
        start = stratum * len(ordered) // min(strata, len(ordered))
        end = (stratum + 1) * len(ordered) // min(strata, len(ordered))
        bucket = ordered[start:end]
        rng.shuffle(bucket)
        quota = (stratum + 1) * sample_size // min(strata, len(ordered)) - stratum * sample_size // min(strata, len(ordered))
        selected.extend(bucket[:quota])
    rng.shuffle(selected)
    return selected


def stratified_split(pairs: Sequence[Pair], seed: int, strata: int = 20) -> tuple[list[Pair], list[Pair], list[Pair]]:
    """Make exact 80/10/10 splits, balanced over source/target mean length."""
    if len(pairs) % 10:
        raise ValueError("--target-size must be divisible by 10 for an exact 80/10/10 split")
    ordered = sorted(pairs, key=lambda pair: (pair.average_length, pair.source, pair.target))
    rng = random.Random(seed)
    shuffled: list[Pair] = []
    stratum_count = min(strata, len(ordered))
    for stratum in range(stratum_count):
        start = stratum * len(ordered) // stratum_count
        end = (stratum + 1) * len(ordered) // stratum_count
        bucket = ordered[start:end]
        rng.shuffle(bucket)
        shuffled.extend(bucket)

    train: list[Pair] = []
    validation: list[Pair] = []
    test: list[Pair] = []
    # Every consecutive ten positions contain eight train, one validation, and
    # one test example.  Since each stratum was shuffled, every split retains
    # the same broad length distribution.
    for index, pair in enumerate(shuffled):
        remainder = index % 10
        if remainder < 8:
            train.append(pair)
        elif remainder == 8:
            validation.append(pair)
        else:
            test.append(pair)
    return train, validation, test


def write_json(path: Path, records: Sequence[Pair]) -> None:
    path.write_text(json.dumps([record.as_record() for record in records], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Pair]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.as_record(), ensure_ascii=False) + "\n")


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(records: Sequence[Pair]) -> dict[str, float | int]:
    return {
        "pairs": len(records),
        "average_source_chars": round(statistics.mean(len(record.source) for record in records), 2),
        "average_target_chars": round(statistics.mean(len(record.target) for record in records), 2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Helsinki-NLP/opus_books")
    parser.add_argument("--config", default="de-en", help="Hugging Face OPUS Books language-pair configuration")
    parser.add_argument("--split", default="train", help="Source split in the Hugging Face dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--name", default="opus_de_en_25k", help="Base name for emitted files")
    parser.add_argument("--target-size", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-limit", type=int, help="Read only this many source rows; useful for smoke tests")
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=200)
    parser.add_argument("--min-length-ratio", type=float, default=0.5)
    parser.add_argument("--max-length-ratio", type=float, default=2.0)
    parser.add_argument("--language-confidence", type=float, default=0.90)
    parser.add_argument("--skip-language-id", action="store_true")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.95)
    parser.add_argument("--discard-lowest-quality", type=float, default=0.10)
    parser.add_argument(
        "--quality-model",
        default="distilgpt2",
        help="Causal English model for target perplexity, or 'none' to skip it",
    )
    parser.add_argument("--quality-batch-size", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.target_size <= 0 or args.target_size % 10:
        raise ValueError("--target-size must be a positive multiple of 10")
    if not 0 <= args.discard_lowest_quality < 1:
        raise ValueError("--discard-lowest-quality must be in [0, 1)")
    if not 0 < args.fuzzy_threshold <= 1:
        raise ValueError("--fuzzy-threshold must be in (0, 1]")
    if not args.skip_language_id and detect_langs is None:
        raise RuntimeError(
            "Language identification requires langdetect. Install it with "
            "`pip install langdetect`, or use --skip-language-id for a smoke test."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    raw_pairs: list[Pair] = []
    for pair in load_raw_pairs(args.dataset, args.config, args.split, args.raw_limit):
        counters["raw_rows"] += 1
        if is_clean_pair(pair, args, counters):
            raw_pairs.append(pair)
    LOGGER.info("Retained %s/%s after validation", len(raw_pairs), counters["raw_rows"])

    deduplicated = deduplicate(raw_pairs, args.fuzzy_threshold, counters)
    LOGGER.info("Retained %s after deduplication", len(deduplicated))
    model_name = None if args.quality_model.lower() == "none" else args.quality_model
    scored = score_pairs(deduplicated, model_name, args.quality_batch_size)
    quality_filtered = keep_top_quality(scored, args.discard_lowest_quality)
    counters["low_quality"] = len(scored) - len(quality_filtered)
    selected = stratified_sample(quality_filtered, args.target_size, args.seed)
    train, validation, test = stratified_split(selected, args.seed)

    outputs = {
        "combined_json": (args.output_dir / f"{args.name}.json", selected),
        "train_json": (args.output_dir / f"{args.name}_train.json", train),
        "validation_json": (args.output_dir / f"{args.name}_val.json", validation),
        "test_json": (args.output_dir / f"{args.name}_test.json", test),
        "combined_jsonl": (args.output_dir / f"{args.name}.jsonl", selected),
        "train_jsonl": (args.output_dir / f"{args.name}_train.jsonl", train),
        "validation_jsonl": (args.output_dir / f"{args.name}_val.jsonl", validation),
        "test_jsonl": (args.output_dir / f"{args.name}_test.jsonl", test),
    }
    for label, (path, records) in outputs.items():
        (write_json if label.endswith("json") else write_jsonl)(path, records)

    metadata_path = args.output_dir / f"{args.name}_metadata.json"
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"dataset": args.dataset, "config": args.config, "split": args.split},
        "license_note": (
            "OPUS Books permits personal, educational, and research use; review "
            "the upstream dataset card before commercial use or redistribution."
        ),
        "seed": args.seed,
        "filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "length_ratio": [args.min_length_ratio, args.max_length_ratio],
            "language_confidence": None if args.skip_language_id else args.language_confidence,
            "fuzzy_duplicate_threshold": args.fuzzy_threshold,
            "discard_lowest_quality_fraction": args.discard_lowest_quality,
            "quality_model": model_name,
        },
        "filter_counts": dict(sorted(counters.items())),
        "splits": {
            "train": describe(train),
            "validation": describe(validation),
            "test": describe(test),
            "total": describe(selected),
        },
        "files": {
            label: {"path": str(path), "md5": md5sum(path), "records": len(records)}
            for label, (path, records) in outputs.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s high-quality pairs to %s", len(selected), args.output_dir)
    LOGGER.info("Split sizes: train=%s validation=%s test=%s", len(train), len(validation), len(test))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
