#!/usr/bin/env python3
"""
NeSyKGLLM - Step 2: Fine-tune WITHOUT Rules
==============================================
Fine-tunes the model on training data generated WITHOUT symbolic rule context,
then evaluates on the test set.

Usage:
    python step2_finetune_no_rules.py --config config.json
"""

import argparse
import os
from functools import partial

import pandas as pd
import torch
from datasets import Dataset

from utils import (
    load_config,
    set_all_seeds,
    login_huggingface,
    create_bnb_config,
    load_model,
    preprocess_batch,
    fine_tune_model,
    evaluate_model,
    print_results,
    save_results_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="NeSyKGLLM - Fine-tune WITHOUT Rules"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg["seed"])
    login_huggingface(cfg["huggingface_token"])

    output_dir = cfg["output_dir"]
    eval_cfg = cfg.get("evaluation", {})
    tcfg = cfg.get("training", {})
    model_tag = cfg["model_key"].replace("-", "_")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    pregen = cfg.get("pregenerated_data", {})
    if pregen.get("use_pregenerated") and pregen.get("train_without_rules_csv"):
        train_path = pregen["train_without_rules_csv"]
    else:
        train_path = os.path.join(output_dir, "train_data_without_rules.csv")

    if pregen.get("use_pregenerated") and pregen.get("test_shared_baseline_csv"):
        test_path = pregen["test_shared_baseline_csv"]
    else:
        test_path = os.path.join(output_dir, "test_shared_baseline.csv")

    print(f"Loading training data: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data:     {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: FINE-TUNING WITHOUT RULES")
    print("=" * 70)
    print(f"Model: {cfg['model_name']}")

    bnb_config = create_bnb_config(cfg)
    model, tokenizer = load_model(
        cfg["model_name"], bnb_config, cfg.get("gpu_max_memory_mb", 40960)
    )

    # ------------------------------------------------------------------
    # Prepare dataset
    # ------------------------------------------------------------------
    train_dataset = Dataset.from_pandas(train_df)
    max_length = tcfg.get("max_length", 512)
    train_dataset = train_dataset.map(
        lambda batch: preprocess_batch(batch, tokenizer, max_length=max_length),
        batched=True,
    )

    # ------------------------------------------------------------------
    # Fine-tune
    # ------------------------------------------------------------------
    model_output_dir = os.path.join(output_dir, f"finetuned_{model_tag}_baseline")
    model = fine_tune_model(model, tokenizer, train_dataset, model_output_dir, cfg)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print("\nEVALUATING FINE-TUNED MODEL (WITHOUT RULES)")
    results = evaluate_model(
        model,
        tokenizer,
        test_df,
        max_samples=eval_cfg.get("max_samples", 2000),
        max_new_tokens=eval_cfg.get("max_new_tokens", 150),
    )
    print_results("FINE-TUNED MODEL (WITHOUT RULES) RESULTS:", results)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = os.path.join(output_dir, "finetune_no_rules_results.json")
    save_results_json(results_path, {
        "model": cfg["model_key"],
        "model_name": cfg["model_name"],
        "step": "finetune_without_rules",
        "model_saved_to": model_output_dir,
        "metrics": results,
    })

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print("\nFine-tuning (without rules) complete!")


if __name__ == "__main__":
    main()
