#!/usr/bin/env python3
"""
NeSyKGLLM - Step 3: Fine-tune WITH Rules
==========================================
Fine-tunes the model on training data generated WITH symbolic rule context.
Runs SEPARATE fine-tuning for each CoT version:
  - CoT2 (POSITIVE/NEGATIVE only)             -> NeSyKGLLM version 1
  - CoT3 (VALID/INVALID + POSITIVE/NEGATIVE)  -> NeSyKGLLM version 2

Each produces its own model checkpoint and evaluation results,
enabling direct comparison.

Usage:
    python step3_finetune_with_rules.py --config config.json
    python step3_finetune_with_rules.py --config config.json --cot_version CoT2
    python step3_finetune_with_rules.py --config config.json --cot_version CoT3
"""

import argparse
import os

import gc

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import WeightedRandomSampler

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


def _build_weighted_sampler(train_df: pd.DataFrame):
    """Build WeightedRandomSampler from 'weight' column if present and varied."""
    if "weight" not in train_df.columns:
        return None
    weights = train_df["weight"].fillna(1.0).tolist()
    if len(set(weights)) <= 1:
        return None
    print(f"  [Method 4] WeightedRandomSampler: "
          f"min={min(weights):.2f}, max={max(weights):.2f}, "
          f"mean={sum(weights)/len(weights):.2f}")
    return WeightedRandomSampler(weights=weights, num_samples=len(weights),
                                 replacement=True)


def run_finetune_for_cot_version(cot_version, cfg, train_path):
    """
    Run a complete fine-tune + evaluate cycle for one CoT version.

    Args:
        cot_version: "CoT2" or "CoT3"
        cfg: loaded config dict
        train_path: path to TRAINING CSV for this CoT version

    Each variant is evaluated on its own shared test CSV (same entity instances,
    format-appropriate symbolic context).
    """
    output_dir = cfg["output_dir"]
    eval_cfg = cfg.get("evaluation", {})
    tcfg = cfg.get("training", {})
    model_tag = cfg["model_key"].replace("-", "_")

    print(f"\n{'=' * 70}")
    print(f"STEP 3: FINE-TUNING WITH RULES — {cot_version}")
    if cot_version == "CoT2":
        print("  NeSyKGLLM Version 1: POSITIVE/NEGATIVE classification")
    else:
        print("  NeSyKGLLM Version 2: VALID/INVALID + POSITIVE/NEGATIVE")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading training data: {train_path}")
    train_df = pd.read_csv(train_path)

    # Subsample to exactly what the trainer will use.
    # With max_steps=N and batch_size=1, only N rows are ever seen.
    # Loading all 400k+ rows then tokenizing them wastes ~5GB RAM and
    # causes OOM before the model is even loaded.
    num_steps   = tcfg.get("num_steps", 20000)
    batch_size  = tcfg.get("per_device_train_batch_size", 1)
    grad_accum  = tcfg.get("gradient_accumulation_steps", 8)
    rows_needed = num_steps * batch_size * grad_accum   # effective steps * samples/step
    # Add 20% buffer so the sampler has room to shuffle without running dry
    rows_to_load = min(len(train_df), int(rows_needed * 1.2))
    n_total = len(train_df)
    if rows_to_load < n_total:
        train_df = train_df.sample(n=rows_to_load, random_state=cfg.get("seed", 42)) \
                           .reset_index(drop=True)
        print(f"  Subsampled training data: {rows_to_load:,} rows "
              f"(from {n_total:,} total, needed ~{rows_needed:,} for {num_steps} steps)")
    else:
        print(f"  Using full training set: {n_total:,} rows")

    # Each variant uses its own format of the shared test set
    shared_test_path = resolve_shared_test_path(cfg, cot_version=cot_version)
    print(f"Loading shared test data: {shared_test_path}")
    test_df = pd.read_csv(shared_test_path)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)} (shared)")

    # Print distribution info
    if "classification" in train_df.columns:
        print(f"  Classification: {dict(train_df['classification'].value_counts())}")
    if "validity" in train_df.columns:
        vc = train_df["validity"].dropna().value_counts()
        if len(vc) > 0:
            print(f"  Validity: {dict(vc)}")

    # Build sampler BEFORE dropping the weight column
    sampler = _build_weighted_sampler(train_df)

    # ------------------------------------------------------------------
    # Load model (fresh for each CoT version)
    # ------------------------------------------------------------------
    print(f"\nModel: {cfg['model_name']}")
    bnb_config = create_bnb_config(cfg)
    model, tokenizer = load_model(
        cfg["model_name"], bnb_config, cfg.get("gpu_max_memory_mb", 40960)
    )

    # ------------------------------------------------------------------
    # Prepare dataset — drop weight column, it is not a model input
    # ------------------------------------------------------------------
    train_df_hf = train_df.drop(columns=["weight"], errors="ignore")
    train_dataset = Dataset.from_pandas(train_df_hf)
    max_length = tcfg.get("max_length", 512)
    train_dataset = train_dataset.map(
        lambda batch: preprocess_batch(batch, tokenizer, max_length=max_length),
        batched=True,
    )

    # ------------------------------------------------------------------
    # Fine-tune
    # ------------------------------------------------------------------
    model_output_dir = os.path.join(
        output_dir, f"finetuned_{model_tag}_with_rules_{cot_version}"
    )
    model = fine_tune_model(
        model, tokenizer, train_dataset, model_output_dir, cfg,
        sampler=sampler,
    )

    # ------------------------------------------------------------------
    # Evaluate — two passes:
    #   1. Format-matched test set  (sanity check — model sees same symbolic
    #      context it was trained on, confirms fine-tuning worked)
    #   2. Clean eval set           (fair cross-format comparison — NO rule
    #      context, NO tags; same file used for Baseline, CoT2, CoT3)
    # ------------------------------------------------------------------
    eval_clean_path = os.path.join(output_dir, "test_eval_clean.csv")
    if not os.path.exists(eval_clean_path):
        raise FileNotFoundError(
            f"Clean eval file not found: {eval_clean_path}\n"
            "Run prepare_data.py first to generate test_eval_clean.csv."
        )
    eval_clean_df = pd.read_csv(eval_clean_path)

    print(f"\nEVALUATING (format-matched) — {cot_version}")
    results_matched = evaluate_model(
        model, tokenizer, test_df,
        max_samples=eval_cfg.get("max_samples", 2000),
        max_new_tokens=eval_cfg.get("max_new_tokens", 150),
    )
    print_results(f"Format-matched results ({cot_version}):", results_matched)

    print(f"\nEVALUATING (clean, cross-format) — {cot_version}")
    results_clean = evaluate_model(
        model, tokenizer, eval_clean_df,
        max_samples=eval_cfg.get("max_samples", 2000),
        max_new_tokens=eval_cfg.get("max_new_tokens", 150),
    )
    print_results(f"Clean eval results ({cot_version}):", results_clean)

    # Use clean results as the primary reported metric
    results = results_clean

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = os.path.join(
        output_dir, f"finetune_with_rules_{cot_version}_results.json"
    )
    save_results_json(results_path, {
        "model": cfg["model_key"],
        "model_name": cfg["model_name"],
        "step": f"finetune_with_rules_{cot_version}",
        "cot_version": cot_version,
        "model_saved_to": model_output_dir,
        "train_data": train_path,
        "test_data_format_matched": shared_test_path,
        "test_data_clean_eval": eval_clean_path,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "metrics_format_matched": results_matched,
        "metrics_clean_eval": results_clean,
        "metrics": results_clean,   # primary metric = clean cross-format eval
    })

    # Cleanup — force full GPU + CPU memory reclamation before next model load.
    # Without gc.collect(), Python may hold LoRA adapter references in CPU RAM
    # long enough to cause OOM when the next model is loaded.
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\nFine-tuning ({cot_version}) complete!")

    return results


def resolve_shared_test_path(cfg: dict, cot_version: str = "CoT2", fallback: str = None) -> str:
    """
    Resolve the shared test CSV for a specific CoT variant.

    Each variant is evaluated on the same entity instances but with
    format-appropriate symbolic context:
      - Baseline : test_shared_baseline.csv
      - CoT2     : test_shared_CoT2.csv
      - CoT3     : test_shared_CoT3.csv
    """
    pregen = cfg.get("pregenerated_data", {})
    output_dir = cfg.get("output_dir", "./outputs/")

    variant_map = {
        "CoT2":     [
            pregen.get("test_shared_cot2_csv"),
            os.path.join(output_dir, "test_shared_CoT2.csv"),
        ],
        "CoT3":     [
            pregen.get("test_shared_cot3_csv"),
            os.path.join(output_dir, "test_shared_CoT3.csv"),
        ],
        "Baseline": [
            pregen.get("test_shared_baseline_csv"),
            os.path.join(output_dir, "test_shared_baseline.csv"),
        ],
    }

    candidates = variant_map.get(cot_version, []) + [fallback]

    for path in candidates:
        if path and os.path.exists(path):
            return path

    # Return first non-None with warning
    for path in candidates:
        if path:
            print(f"  WARNING: Shared test CSV not found; will attempt: {path}")
            return path

    raise FileNotFoundError(
        f"No shared test CSV found for {cot_version}. "
        "Run prepare_data.py first to generate test_shared_*.csv files."
    )


def resolve_data_paths(cfg, cot_version):
    """
    Resolve TRAINING CSV path for a given CoT version.

    Test path is always resolved separately via resolve_shared_test_path().
    """
    output_dir = cfg["output_dir"]
    pregen = cfg.get("pregenerated_data", {})

    if pregen.get("use_pregenerated"):
        key = f"train_with_rules_{cot_version.lower()}_csv"
        train_path = pregen.get(
            key,
            pregen.get("train_with_rules_csv",
                        os.path.join(output_dir,
                                     f"train_data_with_rules_{cot_version}.csv"))
        )
    else:
        train_path = os.path.join(
            output_dir, f"train_data_with_rules_{cot_version}.csv"
        )

    return train_path


def main():
    parser = argparse.ArgumentParser(
        description="NeSyKGLLM - Fine-tune WITH Rules (CoT2 and/or CoT3)"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config JSON"
    )
    parser.add_argument(
        "--cot_version", type=str, default=None,
        choices=["CoT2", "CoT3", "both"],
        help="Which CoT version to fine-tune. "
             "Default: runs both if CoT3 data exists, otherwise CoT2 only."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg["seed"])
    login_huggingface(cfg["huggingface_token"])

    # --- GPU sanity check before doing anything ---
    import torch as _torch
    n_gpus = _torch.cuda.device_count()
    print(f"\nGPU check: {n_gpus} GPU(s) visible to PyTorch")
    for i in range(n_gpus):
        props = _torch.cuda.get_device_properties(i)
        free, total = _torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {props.name}, "
              f"total={total/1024**3:.1f}GB, free={free/1024**3:.1f}GB")
    if n_gpus == 0:
        raise RuntimeError(
            "No GPUs visible. Check SLURM --gres=gpu flag and CUDA setup."
        )

    # Determine which versions to run
    versions_to_run = []

    if args.cot_version == "both" or args.cot_version is None:
        versions_to_run.append("CoT2")
        cot3_train = resolve_data_paths(cfg, "CoT3")
        if os.path.exists(cot3_train) or cfg.get("rules_dir_cot3"):
            versions_to_run.append("CoT3")
        elif args.cot_version == "both":
            print("WARNING: --cot_version=both but CoT3 data not found. "
                  "Running CoT2 only.")
    else:
        versions_to_run.append(args.cot_version)

    print(f"Will fine-tune for: {', '.join(versions_to_run)}")

    # Run fine-tuning for each version
    all_results = {}
    for version in versions_to_run:
        train_path = resolve_data_paths(cfg, version)

        if not os.path.exists(train_path):
            print(f"\nERROR: Training data not found: {train_path}")
            print("Run prepare_data.py first to generate the data.")
            continue

        results = run_finetune_for_cot_version(version, cfg, train_path)
        all_results[version] = results

    # ------------------------------------------------------------------
    # Comparison summary (if both were run)
    # ------------------------------------------------------------------
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("COMPARISON: CoT2 vs CoT3  (clean cross-format eval — no tags)")
        print(f"{'=' * 70}")
        print(f"{'Metric':<15} {'CoT2 (v1)':>12} {'CoT3 (v2)':>12} {'Delta':>10}")
        print("-" * 50)
        for metric in ["accuracy", "f1_score", "precision", "recall"]:
            v2 = all_results.get("CoT2", {}).get(metric, 0)
            v3 = all_results.get("CoT3", {}).get(metric, 0)
            delta = v3 - v2
            sign = "+" if delta >= 0 else ""
            print(f"{metric:<15} {v2:>12.4f} {v3:>12.4f} {sign}{delta:>9.4f}")

        comp_path = os.path.join(cfg["output_dir"], "comparison_CoT2_vs_CoT3.json")
        save_results_json(comp_path, {
            "evaluation": "clean_cross_format",
            "eval_file": "test_eval_clean.csv",
            "note": "All models evaluated on path facts + question only — "
                    "no rule context, no symbolic tags visible to the model.",
            "CoT2_metrics": {
                k: v for k, v in all_results.get("CoT2", {}).items()
                if k not in ("y_true", "y_pred")
            },
            "CoT3_metrics": {
                k: v for k, v in all_results.get("CoT3", {}).items()
                if k not in ("y_true", "y_pred")
            },
        })

    print("\nAll fine-tuning complete!")


if __name__ == "__main__":
    main()
