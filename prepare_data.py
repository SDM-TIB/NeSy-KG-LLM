#!/usr/bin/env python3
"""
NeSyKGLLM - Data Preparation
==============================
Preprocesses the knowledge graph and generates training/test data
from pre-generated CoT rule files.

Data split strategy:
  - Rule-based (CoT2/CoT3): ALL instances are pooled from rule files,
    then split 80:20 with stratified balancing. No user-specified sample
    counts — sizes are determined by available instances.
  - Baseline (no rules): Random walk with user-specified total_samples,
    then split 80:20.

Generates SEPARATE datasets for each experiment:
  1. train_data_without_rules.csv / test_data_without_rules.csv
  2. train_data_with_rules_CoT2.csv / test_data_CoT2.csv
  3. train_data_with_rules_CoT3.csv / test_data_CoT3.csv

Config expects two rule directories:
  "rules_dir_cot2": "path/to/cot2_rules/"
  "rules_dir_cot3": "path/to/cot3_rules/"

Usage:
    python prepare_data.py --config config.json
"""

import argparse
import json
import os
import re
import shutil
from collections import Counter

import pandas as pd

from utils import (
    load_config,
    set_all_seeds,
    preprocess_kg_file,
    generate_relation2id,
    load_knowledge_graph,
    load_relation_mapping,
    load_all_rules,
    generate_shared_test_set,
    generate_training_data_with_rules,
)


def _generate_and_save(label, graph, node_list, relation2id, rules,
                       dg, pca_threshold, output_dir, use_rules,
                       train_filename,
                       negative_mining=None,
                       excluded_keys=None):
    """
    Generate a training dataset and save CSV.

    excluded_keys: set of (rule_id, instance_text) tuples that belong to the
                   shared test set and must be excluded from training data.
    """
    train_ratio = dg.get("train_ratio", 0.8)
    total_samples = dg.get("baseline_samples", 2000) if not use_rules else None

    train_df, _ = generate_training_data_with_rules(
        graph, node_list, relation2id, rules,
        max_path_length=dg["max_path_length"],
        include_reasoning=dg["include_reasoning"],
        use_rules=use_rules,
        max_rules_in_context=dg.get("max_rules_in_context", 3),
        pca_threshold=pca_threshold,
        train_ratio=train_ratio,
        total_samples=total_samples,
        negative_mining=negative_mining,
        excluded_keys=excluded_keys,
        imbalance_threshold=dg.get("imbalance_threshold", 1.5),
    )

    train_path = os.path.join(output_dir, train_filename)
    train_df.to_csv(train_path, index=False)

    print(f"\n  {label}:")
    print(f"    Train: {len(train_df)} samples -> {train_path}")

    # Use Label column (canonical ground truth, always present in generated CSVs).
    # output_text string matching is unreliable — reasoning text contains
    # "yes"/"no" throughout, causing overcounting.
    if "Label" in train_df.columns:
        n_yes = (train_df["Label"] == 1).sum()
        n_no  = (train_df["Label"] == 0).sum()
        print(f"    Train: yes={n_yes}, no={n_no} (from Label col)")
    else:
        n_yes = train_df["output_text"].str.contains(r"The answer is yes", case=False, na=False, regex=True).sum()
        n_no  = train_df["output_text"].str.contains(r"The answer is no",  case=False, na=False, regex=True).sum()
        print(f"    Train: yes={n_yes}, no={n_no} (from output_text fallback)")

    return train_df


def _get_first_relation(text: str) -> str:
    """Extract the first relation token from an input_text string."""
    m = re.search(r'\bhas (\w+)\b', str(text))
    return m.group(1) if m else 'unknown'


def filter_skewed_relations(output_dir: str, threshold: float) -> set:
    """
    Detect and remove relations with >threshold% label skew from ALL
    generated CSVs in output_dir.

    Called automatically by prepare_data.py when
    data_generation.relation_skew_threshold is set in config.json.
    Only uses the CoT2 training set to decide which relations to filter --
    the same set is then applied to every other CSV for consistency.

    Backs up originals as *_unfiltered.csv before overwriting.
    Saves filtered_relations.json for reproducibility.

    Returns the set of filtered relation names (empty set = nothing filtered).
    """
    train_path = os.path.join(output_dir, 'train_data_with_rules_CoT2.csv')
    # Always compute skew from the ORIGINAL unfiltered data so that
    # re-runs at different thresholds see the full relation distribution
    unfiltered_path = train_path.replace('.csv', '_unfiltered.csv')
    source_path = unfiltered_path if os.path.exists(unfiltered_path) else train_path
    if not os.path.exists(source_path):
        print(f"  WARNING: {source_path} not found -- skipping relation filter.")
        return set()

    print(f"\n{'=' * 70}")
    print(f"RELATION SKEW FILTER (threshold: >{threshold}% label dominance)")
    print(f"{'=' * 70}")
    print(f"  Computing skew from: {source_path}")

    # Load only the two columns needed to save RAM on large datasets
    train_df = pd.read_csv(source_path, usecols=['input_text', 'Label'])
    train_df['_rel'] = train_df['input_text'].apply(_get_first_relation)

    # Compute per-relation label distribution
    stats = []
    for rel, group in train_df.groupby('_rel'):
        n_yes   = (group['Label'] == 1).sum()
        n_no    = (group['Label'] == 0).sum()
        total   = len(group)
        yes_pct = n_yes / total * 100
        no_pct  = n_no  / total * 100
        stats.append((rel, yes_pct, no_pct, total))

    del train_df  # free RAM immediately

    skewed = {r for r, y, n, _ in stats if max(y, n) >= threshold}
    total_samples  = sum(c for _, _, _, c in stats)
    skewed_samples = sum(c for r, _, _, c in stats if r in skewed)

    print(f"  Relations analysed:      {len(stats)}")
    print(f"  Skewed (>={threshold}%): {len(skewed)} relations, "
          f"{skewed_samples:,} samples ({skewed_samples/total_samples*100:.1f}%)")

    if not skewed:
        print(f"  No skewed relations found. Dataset is clean.")
        return set()

    print(f"  Top skewed relations by sample count:")
    skewed_stats = sorted(
        [(r, y, n, c) for r, y, n, c in stats if r in skewed],
        key=lambda x: x[3], reverse=True
    )
    for r, y, n, c in skewed_stats[:10]:
        dominant = f"{max(y,n):.0f}% {'YES' if y > n else 'NO'}"
        print(f"    {r:<25} {dominant}  ({c:,} samples)")
    if len(skewed_stats) > 10:
        print(f"    ... and {len(skewed_stats)-10} more")

    # Apply the same filter set to all relevant CSVs.
    # IMPORTANT: always restore from *_unfiltered.csv backup first so that
    # re-runs at a different threshold always filter the original data,
    # never a previously filtered version.
    csv_files = [
        'train_data_with_rules_CoT2.csv',
        'train_data_with_rules_CoT3.csv',
        'train_data_without_rules.csv',
        'test_shared_CoT2.csv',
        'test_shared_CoT3.csv',
        'test_shared_baseline.csv',
        'test_eval_clean.csv',
    ]

    print(f"\n  Restoring originals from backups (if present)...")
    for fname in csv_files:
        fpath  = os.path.join(output_dir, fname)
        backup = fpath.replace('.csv', '_unfiltered.csv')
        if os.path.exists(backup):
            shutil.copy(backup, fpath)
            print(f"    Restored: {fname}")

    print(f"\n  Filtering CSVs in {output_dir} (threshold={threshold}%):")
    for fname in csv_files:
        fpath = os.path.join(output_dir, fname)
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        if 'input_text' not in df.columns:
            continue
        original = len(df)
        df['_rel'] = df['input_text'].apply(_get_first_relation)
        filtered  = df[~df['_rel'].isin(skewed)].drop(columns=['_rel'])
        removed   = original - len(filtered)

        # Create backup from original if not already present
        backup = fpath.replace('.csv', '_unfiltered.csv')
        if not os.path.exists(backup):
            shutil.copy(fpath, backup)

        filtered.to_csv(fpath, index=False)
        print(f"    {fname}: {original:,} -> {len(filtered):,} "
              f"(removed {removed:,} = {removed/original*100:.1f}%)")

    # Save filter log for reproducibility / paper methods section
    log_path = os.path.join(output_dir, 'filtered_relations.json')
    with open(log_path, 'w') as f:
        json.dump({
            'threshold': threshold,
            'n_skewed_relations': len(skewed),
            'skewed_relations': sorted(skewed),
            'relation_stats': [
                {'relation': r, 'yes_pct': round(y, 2), 'no_pct': round(n, 2),
                 'count': c, 'filtered': r in skewed}
                for r, y, n, c in sorted(stats, key=lambda x: x[3], reverse=True)
            ]
        }, f, indent=2)
    print(f"\n  Filter log saved: {log_path}")
    print(f"  Originals backed up as *_unfiltered.csv")
    return skewed


def main():
    parser = argparse.ArgumentParser(description="NeSyKGLLM Data Preparation")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg["seed"])

    data_dir = cfg["data_dir"]
    output_dir = cfg["output_dir"]
    kg = cfg["kg_files"]
    dg = cfg["data_generation"]
    pca_threshold = dg.get("pca_threshold", 0.5)

    # ------------------------------------------------------------------
    # 1. Preprocess KG (optional)
    # ------------------------------------------------------------------
    train_raw = os.path.join(data_dir, kg["train"])
    train_processed = os.path.join(data_dir, kg["train_processed"])
    relation2id_path = os.path.join(data_dir, kg["relation2id"])

    if cfg.get("preprocess_kg", False):
        print("=" * 70)
        print("PREPROCESSING KNOWLEDGE GRAPH")
        print("=" * 70)
        preprocess_kg_file(train_raw, train_processed)
        generate_relation2id(train_processed, relation2id_path)

    # ------------------------------------------------------------------
    # 2. Load KG
    # ------------------------------------------------------------------
    print("=" * 70)
    print("LOADING KNOWLEDGE GRAPH")
    print("=" * 70)

    graph, node_list = load_knowledge_graph(train_processed)
    relation2id = load_relation_mapping(relation2id_path)
    print(f"Graph loaded: {len(graph)} nodes, {len(node_list)} unique entities")
    print(f"PCA confidence threshold: {pca_threshold}")

    # ------------------------------------------------------------------
    # 3. Load rules from BOTH CoT2 and CoT3 directories
    # ------------------------------------------------------------------
    rules_dir_cot2 = cfg.get("rules_dir_cot2", cfg.get("rules_dir"))
    rules_dir_cot3 = cfg.get("rules_dir_cot3")

    print("\n" + "=" * 70)
    print("LOADING CoT2 RULES")
    print("=" * 70)
    rules_cot2 = load_all_rules(rules_dir_cot2)
    n2 = sum(len(r["instances"]) for r in rules_cot2)
    print(f"CoT2: {len(rules_cot2)} rules, {n2} total instances")

    rules_cot3 = []
    if rules_dir_cot3:
        print("\n" + "=" * 70)
        print("LOADING CoT3 RULES")
        print("=" * 70)
        rules_cot3 = load_all_rules(rules_dir_cot3)
        n3 = sum(len(r["instances"]) for r in rules_cot3)
        print(f"CoT3: {len(rules_cot3)} rules, {n3} total instances")
    else:
        print("\nNo rules_dir_cot3 in config — skipping CoT3 generation.")

    # ------------------------------------------------------------------
    # 4. Generate shared test set from CoT3 instances (all three formats)
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    negative_mining_cfg = cfg.get("negative_mining", None)
    if negative_mining_cfg and negative_mining_cfg.get("enable"):
        print(f"\n  Negative mining config: {negative_mining_cfg}")

    if rules_cot3:
        print("\n" + "=" * 70)
        print("GENERATING: Shared test set (Baseline / CoT2 / CoT3 formats)")
        print("=" * 70)
        train_keys, test_keys, test_baseline_df, test_cot2_df, test_cot3_df, test_eval_df = \
            generate_shared_test_set(
                rules_cot3=rules_cot3,
                rules_cot2=rules_cot2,
                train_ratio=dg.get("train_ratio", 0.8),
                include_reasoning=dg["include_reasoning"],
                imbalance_threshold=dg.get("imbalance_threshold", 1.5),
            )
        test_baseline_df.to_csv(os.path.join(output_dir, "test_shared_baseline.csv"), index=False)
        test_cot2_df.to_csv(    os.path.join(output_dir, "test_shared_CoT2.csv"),     index=False)
        test_cot3_df.to_csv(    os.path.join(output_dir, "test_shared_CoT3.csv"),     index=False)
        test_eval_df.to_csv(    os.path.join(output_dir, "test_eval_clean.csv"),      index=False)
        print(f"  Saved: test_shared_baseline.csv ({len(test_baseline_df)} samples)")
        print(f"  Saved: test_shared_CoT2.csv     ({len(test_cot2_df)} samples)")
        print(f"  Saved: test_shared_CoT3.csv     ({len(test_cot3_df)} samples)")
        print(f"  Saved: test_eval_clean.csv      ({len(test_eval_df)} samples)  <-- cross-format eval")
    else:
        print("\nNo CoT3 rules — skipping shared test set.")
        train_keys = test_keys = None

    # ------------------------------------------------------------------
    # 5. Generate training data (excluding shared test instances)
    # ------------------------------------------------------------------

    # --- Baseline: without rules (random walk) ---
    print("\n" + "=" * 70)
    print("GENERATING: Baseline training data WITHOUT rules (random walk)")
    print("=" * 70)
    _generate_and_save(
        "Baseline (no rules)", graph, node_list, relation2id, rules_cot2,
        dg, pca_threshold, output_dir,
        use_rules=False,
        train_filename="train_data_without_rules.csv",
        negative_mining=None,
        excluded_keys=None,   # Random walk — no overlap possible
    )

    # --- CoT2: with rules ---
    print("\n" + "=" * 70)
    print("GENERATING: CoT2 training data (POSITIVE/NEGATIVE)")
    print("=" * 70)
    _generate_and_save(
        "CoT2 (with rules)", graph, node_list, relation2id, rules_cot2,
        dg, pca_threshold, output_dir,
        use_rules=True,
        train_filename="train_data_with_rules_CoT2.csv",
        negative_mining={
            "enable": negative_mining_cfg.get("enable", False),
            "enable_weighting": False,
            "base_weight": negative_mining_cfg.get("base_weight", 1.0),
            "max_weight": 1.0,
        } if negative_mining_cfg else None,
        excluded_keys=test_keys,   # Exclude shared test instances
    )

    # --- CoT3: with rules ---
    if rules_cot3:
        print("\n" + "=" * 70)
        print("GENERATING: CoT3 training data (VALID/INVALID + POSITIVE/NEGATIVE)")
        print("=" * 70)
        _generate_and_save(
            "CoT3 (with rules)", graph, node_list, relation2id, rules_cot3,
            dg, pca_threshold, output_dir,
            use_rules=True,
            train_filename="train_data_with_rules_CoT3.csv",
            negative_mining=negative_mining_cfg,
            excluded_keys=test_keys,   # Exclude shared test instances
        )

    # ------------------------------------------------------------------
    # 5. Relation skew filter (optional — controlled by config)
    # ------------------------------------------------------------------
    # Runs AFTER all CSVs are generated so it can compute skew from
    # real label distributions. Applies the same filter set to every
    # CSV so train/test distributions stay consistent.
    # Enable by setting data_generation.relation_skew_threshold in config.
    skew_threshold = dg.get("relation_skew_threshold", None)
    if skew_threshold is not None:
        filter_skewed_relations(output_dir, threshold=float(skew_threshold))
    else:
        print("\n  Relation skew filter: disabled "
              "(set data_generation.relation_skew_threshold in config to enable)")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("DATA GENERATION COMPLETE — OUTPUT FILES")
    print(f"{'=' * 70}")
    import pandas as pd
    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith(".csv"):
            fpath = os.path.join(output_dir, fname)
            df = pd.read_csv(fpath)
            parts = [f"{len(df)} samples"]
            # Use Label column (canonical) when available, fall back to output_text
            if "Label" in df.columns:
                n_yes = (df["Label"] == 1).sum()
                n_no  = (df["Label"] == 0).sum()
                parts.append(f"yes={n_yes} no={n_no} (from Label col)")
            elif "output_text" in df.columns:
                n_yes = df["output_text"].str.contains("yes", case=False, na=False).sum()
                n_no  = df["output_text"].str.contains("The answer is no", case=False, na=False).sum()
                parts.append(f"yes={n_yes} no={n_no} (from output_text)")
            print(f"  {fname}: {', '.join(parts)}")
    print(f"  Columns: {list(df.columns)}")

    print(f"\nAll files saved to: {output_dir}")


if __name__ == "__main__":
    main()
