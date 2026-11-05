# Paragraph-aligned NLLB fine-tuning

Fine-tune [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) for machine translation using paragraph-aligned parallel text.

This repository includes a training script, an inference notebook, and JSON data splits for German-to-English and English-to-Spanish experiments.

## Repository contents

- `scripts/finetune_nllb.py` — command-line training script built with Hugging Face Transformers.
- `scripts/inference.ipynb` — notebook for running inference with a fine-tuned model.
- `data/` — paragraph-aligned JSON datasets. Each item contains `source`, `target`, language labels, and NLLB language codes.

The `opus_de_en_25k_*` files contain German-to-English data; the `opus_test_*` files contain the smaller experiment splits.

## Setup

Use Python 3.10+ and install the dependencies:

```bash
pip install torch transformers datasets evaluate sacrebleu sentencepiece accelerate
```

A CUDA-capable GPU is recommended. The training script uses bfloat16 by default when mixed precision is enabled.

## Fine-tune German to English

```bash
python scripts/finetune_nllb.py \
  --train_dataset data/opus_de_en_25k_train.json \
  --val_dataset data/opus_de_en_25k_val.json \
  --src_lang deu_Latn \
  --tgt_lang eng_Latn \
  --output_dir models/nllb-finetuned-de-en \
  --epochs 3 \
  --batch_size 8 \
  --gradient_accumulation_steps 1
```

Use `--no_fp16` to turn off fp16. Adjust the batch size and gradient accumulation for the available GPU memory.

## Rebuild the OPUS Books dataset

The reproducible German-to-English dataset builder is at `scripts/prepare_opus_dataset.py`. It downloads the `de-en` configuration of `Helsinki-NLP/opus_books`, validates and filters pairs, performs exact and fuzzy deduplication, removes the lowest-quality 10%, and creates balanced 80/10/10 length-stratified splits.

```bash
pip install datasets langdetect rapidfuzz transformers torch
python scripts/prepare_opus_dataset.py --output-dir data
```

It writes the existing JSON-array files used by the training script, matching JSONL files, and `opus_de_en_25k_metadata.json` with filter counts, split statistics, and MD5 checksums. The default quality stage scores English targets with `distilgpt2`; use `--quality-model none` only for a fast development run.

## Dataset format

The trainer accepts a JSON list whose entries follow this shape:

```json
{
  "source": "Es war ganz unm\u00f6glich, an diesem Tage einen Spaziergang zu machen.",
  "target": "There was no possibility of taking a walk that day.",
  "source_lang": "de",
  "target_lang": "en",
  "src_lang_nllb": "deu_Latn",
  "tgt_lang_nllb": "eng_Latn"
}
```

Only `source` and `target` are required by `scripts/finetune_nllb.py`; pass the matching NLLB language codes with `--src_lang` and `--tgt_lang`.

## Inference

Open `scripts/inference.ipynb`, set the fine-tuned model path, and run the cells to translate text. The notebook is a starting point for loading the model and tokenizer after training.

## License and data

Review the licensing and attribution requirements for any source corpus before redistributing or using it in production.
