#!/usr/bin/env python3
"""
Fine-tune NLLB model on Wikipedia parallel corpus

Usage:
python ~/finetune_nllb.py --train_dataset ~/opus_de_en_25k_train.json --val_dataset ~/opus_de_en_25k_val.json --output_dir ~/models/nllb-finetuned-de-en --src_lang deu_Latn --tgt_lang eng_Latn --epochs n --batch_size m --gradient_accumulation_steps k
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import os

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import Dataset
import evaluate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NLLBFineTuner:
    """Fine-tune NLLB model on parallel corpus"""
    
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "eng_Latn",
        tgt_lang: str = "spa_Latn",
        max_length: int = 512,
    ):
        """
        Args:
            model_name: Base NLLB model to fine-tune
            src_lang: Source language code (NLLB format)
            tgt_lang: Target language code (NLLB format)
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            use_fast=True,
        )
        logger.info(f"Tokenizer loaded successfully")
        
        logger.info(f"Loading model: {model_name}")
        logger.info("This may take a few minutes if downloading for the first time...")
        
        # Check if CUDA is available for better memory management
        device_map = "auto" if torch.cuda.is_available() else None
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        
        # Wait for model to be fully loaded
        if hasattr(self.model, 'hf_device_map'):
            logger.info(f"Model device map: {self.model.hf_device_map}")
        
        logger.info(f"✓ Model loaded successfully: {model_name}")
        logger.info(f"✓ Model type: {type(self.model).__name__}")
        logger.info(f"✓ Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Set the tokenizer language codes
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        
        logger.info(f"✓ Language pair configured: {src_lang} -> {tgt_lang}")
    
    def load_dataset(self, dataset_path: str) -> Dataset:
        """
        Load dataset from JSON file
        
        Args:
            dataset_path: Path to JSON dataset file
            
        Returns:
            HuggingFace Dataset
        """
        logger.info(f"Loading dataset from: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to HuggingFace dataset format
        if isinstance(data, list):
            # Format: [{"source": "...", "target": "..."}, ...]
            dataset_dict = {
                'source': [item['source'] for item in data],
                'target': [item['target'] for item in data]
            }
        elif isinstance(data, dict) and 'translation' in data:
            # HuggingFace format
            dataset_dict = {
                'source': [item['en'] for item in data['translation']],
                'target': [item['es'] for item in data['translation']]
            }
        else:
            raise ValueError("Unsupported dataset format")
        
        dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"Loaded {len(dataset)} examples")
        
        return dataset
    
    def preprocess_function(self, examples):
        """Preprocess examples for training"""
        # Set source language
        self.tokenizer.src_lang = self.src_lang
        
        # Tokenize source
        inputs = self.tokenizer(
            examples['source'],
            max_length=self.max_length,
            truncation=True,
            padding=False
        )
        
        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples['target'],
                max_length=self.max_length,
                truncation=True,
                padding=False
            )
        
        inputs['labels'] = labels['input_ids']
        return inputs
    
    def compute_metrics(self, eval_pred):
        """Compute BLEU score during evaluation"""
        predictions, labels = eval_pred
        
        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels (used for padding)
        labels = [[l for l in label if l != -100] for label in labels]
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute BLEU
        bleu_metric = evaluate.load("bleu")
        result = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )
        
        return {"bleu": result["bleu"]}
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        output_dir: str = "./nllb-finetuned",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        save_steps: int = 1000,
        eval_steps: int = 500,
        logging_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        fp16: bool = False,
        bf16: bool =True,
    ):
        """
        Fine-tune the model
        """
        # Preprocess datasets
        logger.info("Preprocessing training dataset...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        if eval_dataset:
            logger.info("Preprocessing validation dataset...")
            eval_dataset = eval_dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        

        if fp16 and bf16:
            fp16=False

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy= "no",
            eval_steps= None,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            warmup_steps=warmup_steps,
            save_strategy= "no",
            save_total_limit=3,
            logging_steps=logging_steps,
            predict_with_generate=True,
            fp16=fp16,
            bf16=bf16,
            push_to_hub=False,
            report_to=[],  # Disable tensorboard to avoid dependency issues
            load_best_model_at_end=False,
            metric_for_best_model="bleu" if eval_dataset else None,
            greater_is_better=True,
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if eval_dataset else None,
        )
        
        # Verify model is ready
        logger.info("Verifying model is ready...")
        logger.info(f"Model device: {self.model.device}")
        logger.info(f"Model dtype: {self.model.dtype}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Train
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Validation samples: {len(eval_dataset)}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        
        trainer.train()
        
        # Save final model
        logger.info(f"Saving final model to: {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Evaluate if validation set provided
        if eval_dataset:
            logger.info("Running final evaluation...")
            eval_results = trainer.evaluate()
            logger.info(f"Final BLEU score: {eval_results.get('eval_bleu', 'N/A')}")
        
        logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune NLLB model on parallel corpus'
    )
    
    # Dataset arguments
    parser.add_argument(
        '--train_dataset',
        type=str,
        required=True,
        help='Path to training dataset (JSON)'
    )
    parser.add_argument(
        '--val_dataset',
        type=str,
        help='Path to validation dataset (JSON)'
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='facebook/nllb-200-distilled-600M',
        help='Base NLLB model (default: facebook/nllb-200-distilled-600M)'
    )
    parser.add_argument(
        '--src_lang',
        type=str,
        default='eng_Latn',
        help='Source language code (default: eng_Latn for English)'
    )
    parser.add_argument(
        '--tgt_lang',
        type=str,
        default='spa_Latn',
        help='Target language code (default: spa_Latn for Spanish)'
    )
    
    # Training arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./nllb-finetuned-en-es',
        help='Output directory for fine-tuned model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Training batch size (default: 8)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Learning rate (default: 2e-5)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Gradient accumulation steps (default: 1)'
    )
    parser.add_argument(
        '--no_fp16',
        action='store_true',
        help='Disable mixed precision training'
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available. Training will be slow on CPU.")
    
    # Create fine-tuner
    fine_tuner = NLLBFineTuner(
        model_name=args.model,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_length=args.max_length
    )
    
    # Load datasets
    train_dataset = fine_tuner.load_dataset(args.train_dataset)
    val_dataset = fine_tuner.load_dataset(args.val_dataset) if args.val_dataset else None
    
    # Train
    fine_tuner.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=not args.no_fp16,
    )
    
    logger.info("\n" + "="*60)
    logger.info("Fine-tuning complete!")
    logger.info("="*60)
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"\nTo use the fine-tuned model in LinguaHaru:")
    logger.info(f"  export NLLB_MODEL_PATH='{args.output_dir}'")
    logger.info(f"  python app.py")


if __name__ == '__main__':
    main()
