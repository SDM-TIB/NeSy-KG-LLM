"""
NeSyKGLLM - Shared Utilities
=============================
Core functions for knowledge graph loading, symbolic rule parsing,
training data generation, model loading, evaluation, and fine-tuning.

Key design: The rule entailments are pre-computed and stored in rule_*.txt
files (CoT2 or CoT3 format). This module parses those files and uses the
instances directly as training/test data, preserving their POSITIVE/NEGATIVE
and (for CoT3) VALID/INVALID classifications.

Two CoT formats are supported:
  - CoT2 (version 1): POSITIVE/NEGATIVE classification only
  - CoT3 (version 2): [VALID]/[INVALID: ShapeName - description] +
    POSITIVE/NEGATIVE classification (v5 format with rich shape context)

Data split strategy:
  - ALL available CoT instances are pooled from rule files
  - A stratified 80:20 train/test split is applied, balanced on
    classification (CoT2) or classification × validity (CoT3)
  - No user-specified sample counts for rule-based data; sizes are
    determined by the available instances
"""

import json
import os
import re
import random
import numpy as np
import pandas as pd

from pathlib import Path

# ---------------------------------------------------------------------------
# GPU/ML imports — guarded so data-prep-only scripts can run without them.
# ---------------------------------------------------------------------------
try:
    import torch
    import bitsandbytes as bnb
    from sklearn.metrics import (
        f1_score, accuracy_score, precision_score, recall_score,
    )
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        set_seed,
        Trainer,
        TrainingArguments,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoPeftModelForCausalLM,
    )
    _ML_AVAILABLE = True
except ImportError as e:
    _ML_AVAILABLE = False
    print(f"Note: ML libraries not fully available ({e}). "
          f"Data preparation will work, but model training/evaluation "
          f"requires torch, transformers, peft, and bitsandbytes.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and validate the JSON configuration file."""
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model_key = cfg["model_key"]
    if model_key not in cfg["available_models"]:
        raise ValueError(
            f"model_key '{model_key}' not found in available_models. "
            f"Options: {list(cfg['available_models'].keys())}"
        )
    cfg["model_name"] = cfg["available_models"][model_key]

    os.makedirs(cfg["output_dir"], exist_ok=True)
    return cfg


def set_all_seeds(seed: int):
    """Set seeds for reproducibility."""
    if _ML_AVAILABLE:
        set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def login_huggingface(token: str):
    """Login to HuggingFace hub."""
    from huggingface_hub import login
    login(token=token)
    print("Logged in to HuggingFace.")


# ---------------------------------------------------------------------------
# Knowledge Graph I/O
# ---------------------------------------------------------------------------

def preprocess_kg_file(input_file: str, output_file: str):
    """Convert tab-separated KG file to space-separated with line count header."""
    with open(input_file, "r") as f:
        lines = f.readlines()
    with open(output_file, "w") as f:
        f.write(f"{len(lines)}\n")
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                f.write(f"{parts[0]} {parts[1]} {parts[2]}\n")
    print(f"Preprocessed KG: {len(lines)} triples -> {output_file}")


def generate_relation2id(processed_kg_path: str, output_path: str):
    """Extract unique relations and write relation2id mapping."""
    relations = set()
    with open(processed_kg_path, "r") as f:
        n = int(f.readline())
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                relations.add(parts[2])
    with open(output_path, "w") as f:
        f.write(f"{len(relations)}\n")
        for i, rel in enumerate(sorted(relations)):
            f.write(f"relation_{rel}\t{rel}\n")
    print(f"Processed {n} triples with {len(relations)} unique relations -> {output_path}")


def load_knowledge_graph(file_path: str):
    """Load KG from preprocessed file. Returns (graph_dict, node_list)."""
    graph = {}
    nodes = set()
    with open(file_path, "r") as f:
        _ = int(f.readline())
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                node1, node2, relation = parts
                nodes.add(node1)
                nodes.add(node2)
                if node1 not in graph:
                    graph[node1] = {}
                graph[node1][node2] = int(relation)
    return graph, list(nodes)


def load_relation_mapping(file_path: str) -> dict:
    """Load relation2id mapping file. Returns {int_id: relation_name}."""
    relation2id = {}
    with open(file_path, "r") as f:
        _ = int(f.readline())
        for line in f:
            relation, relation_id = line.strip().split("\t")
            relation2id[int(relation_id)] = relation
    return relation2id


# ---------------------------------------------------------------------------
# Symbolic Rule File Parsing (Pre-generated CoTs)
# ---------------------------------------------------------------------------

def parse_rule_file(file_path: str) -> dict:
    """
    Parse a single rule .txt file (CoT2 or CoT3 format) into a structured dict.

    Auto-detects format:
      - CoT3: instances contain [VALID] or [INVALID] tags
      - CoT2: instances have no validity tags

    Returns dict with keys:
      rule_id, rule_text, head, body, instances (list of parsed instance dicts),
      pca_confidence, classification, cot_format
    """
    with open(file_path, "r") as f:
        content = f.read()

    rule_info = {
        "rule_id": None,
        "rule_text": None,
        "head": None,
        "body": None,
        "raw_instances": [],
        "instances": [],
        "pca_confidence": None,
        "classification": None,
        "cot_format": None,
    }

    # Parse rule header
    rule_match = re.search(
        r"Rule (\d+):\s*(.+?)(?=\n\nFormal Rule:)", content, re.DOTALL
    )
    if rule_match:
        rule_info["rule_id"] = rule_match.group(1)
        rule_info["rule_text"] = rule_match.group(2).strip()

    # Parse formal rule
    head_match = re.search(r"Head:\s*(.+)", content)
    body_match = re.search(r"Body:\s*(.+)", content)
    if head_match:
        rule_info["head"] = head_match.group(1).strip()
    if body_match:
        rule_info["body"] = body_match.group(1).strip()

    # Parse instances section
    instances_section = re.search(
        r"Real Instances from Knowledge Graph.*?:\n\n(.+?)(?=\n\nRule Statistics:)",
        content,
        re.DOTALL,
    )
    if instances_section:
        for line in instances_section.group(1).strip().split("\n"):
            line = line.strip()
            if line:
                rule_info["raw_instances"].append(line)

    # Parse rule statistics
    pca_match = re.search(r"PCA Confidence:\s*([\d.]+)", content)
    classification_match = re.search(r"Rule Classification:\s*(\w+)", content)
    if pca_match:
        rule_info["pca_confidence"] = float(pca_match.group(1))
    if classification_match:
        rule_info["classification"] = classification_match.group(1)

    # Auto-detect CoT format — recognises both v4 binary [VALID]/[INVALID]
    # and v5 rich [INVALID: ShapeName - description] tags
    has_validity_tags = any(
        re.search(r'\[VALID\]|\[INVALID[\]:]', inst)
        for inst in rule_info["raw_instances"]
    )
    rule_info["cot_format"] = "CoT3" if has_validity_tags else "CoT2"

    for raw_line in rule_info["raw_instances"]:
        parsed = _parse_instance_line(raw_line, rule_info["cot_format"])
        if parsed:
            rule_info["instances"].append(parsed)

    return rule_info


def _parse_instance_line(line: str, cot_format: str) -> dict:
    """
    Parse a single instance line from a rule file.

    CoT2 format:
      Entity has X, also has Y The path is classified as POSITIVE
      (PCA Confidence 0.9982 >= threshold 0.5)

    CoT3 v4 format (binary tag):
      Entity has X, also has Y. [VALID] The path is classified as POSITIVE
      (PCA Confidence 0.9273 >= threshold 0.5)

    CoT3 v5 format (rich shape tag):
      Entity has X, also has Y. [INVALID: PlayerShape - affiliation mismatch (...)]
      The path is classified as NEGATIVE (PCA Confidence 0.3100 < threshold 0.5)

    Returns dict with:
      instance_text, classification, validity, shape_name, shape_description,
      pca_confidence, pca_threshold
    """
    parsed = {
        "instance_text": line,
        "classification": None,
        "validity": None,
        "shape_name": None,        # NEW: e.g. "PlayerShape"
        "shape_description": None, # NEW: e.g. "affiliation mismatch (playsFor ≠ isAffiliatedTo)"
        "pca_confidence": None,
        "pca_threshold": None,
        "answer": None,            # Ground truth: "yes" or "no"
    }

    # Extract ground truth answer (embedded by updated NL-instances scripts)
    answer_match = re.search(r'Answer:\s*(yes|no)', line, re.IGNORECASE)
    if answer_match:
        parsed["answer"] = answer_match.group(1).lower()

    # Extract classification
    cls_match = re.search(
        r'The path is classified as (POSITIVE|NEGATIVE)', line
    )
    if cls_match:
        parsed["classification"] = cls_match.group(1)

    # Extract validity (CoT3 only) — handles both v4 and v5 tag formats
    if cot_format == "CoT3":
        # v5 rich format: [INVALID: ShapeName - description] or [INVALID: Shape1 - desc; Shape2 - desc]
        rich_invalid_match = re.search(
            r'\[INVALID:\s*([^-\]]+?)\s*-\s*([^\]]+?)\]', line
        )
        # v5 multi-shape (no description): [INVALID: Shape1; Shape2]
        multi_shape_match = re.search(
            r'\[INVALID:\s*([^\]]+)\]', line
        )
        # v4 binary format: [VALID] or [INVALID]
        binary_match = re.search(r'\[(VALID|INVALID)\]', line)

        if rich_invalid_match:
            # First shape name and description in a rich tag
            parsed["validity"] = "INVALID"
            parsed["shape_name"] = rich_invalid_match.group(1).strip()
            parsed["shape_description"] = rich_invalid_match.group(2).strip()
        elif multi_shape_match:
            # INVALID with shape name(s) but no description
            parsed["validity"] = "INVALID"
            # Take the first shape if multiple are semicolon-separated
            first_shape = multi_shape_match.group(1).split(";")[0].strip()
            parsed["shape_name"] = first_shape
        elif binary_match:
            parsed["validity"] = binary_match.group(1)

    # Extract PCA confidence and threshold
    pca_match = re.search(
        r'PCA Confidence ([\d.]+)\s*[<>=]+\s*threshold\s*([\d.]+)', line
    )
    if pca_match:
        parsed["pca_confidence"] = float(pca_match.group(1))
        parsed["pca_threshold"] = float(pca_match.group(2))

    return parsed


def load_all_rules(rules_directory: str) -> list:
    """Load all rule_*.txt files from a directory.

    Rules where no instance has an embedded 'Answer: yes/no' tag are
    skipped entirely — using them would silently fall back to the broken
    POSITIVE→yes / NEGATIVE→no heuristic and corrupt ground truth labels.
    Regenerate those rule files with the updated NL-instances scripts.
    """
    rules = []
    skipped = []
    rule_files = sorted(Path(rules_directory).glob("rule_*.txt"))
    for file_path in rule_files:
        try:
            rule_info = parse_rule_file(str(file_path))

            # --- Guard: skip rules with no Answer: tags ---
            n_with_answer = sum(
                1 for i in rule_info["instances"]
                if i.get("answer") is not None
            )
            if n_with_answer == 0 and len(rule_info["instances"]) > 0:
                skipped.append(file_path.name)
                continue

            rules.append(rule_info)
            n_inst = len(rule_info["instances"])
            n_pos = sum(1 for i in rule_info["instances"]
                        if i["classification"] == "POSITIVE")
            n_neg = sum(1 for i in rule_info["instances"]
                        if i["classification"] == "NEGATIVE")
            text_preview = (rule_info["rule_text"] or "")[:60]
            fmt = rule_info["cot_format"]
            print(
                f"  Loaded {file_path.name} [{fmt}]: {text_preview}... "
                f"({n_inst} instances: {n_pos} pos, {n_neg} neg)"
            )
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")

    if skipped:
        print(f"\n  SKIPPED {len(skipped)} rule files — no 'Answer: yes/no' tags found.")
        print(f"  Regenerate with updated NL-instances scripts:")
        for name in skipped:
            print(f"    {name}")
    print(f"\n  Total loaded: {len(rules)} rules "
          f"({len(skipped)} skipped, {len(rule_files)} total files)")
    return rules



def generate_shared_test_set(
    rules_cot3: list,
    rules_cot2: list,
    train_ratio: float = 0.8,
    include_reasoning: bool = True,
    imbalance_threshold: float = 1.5,
) -> tuple:
    """
    Generate a single shared test set from CoT3 instances, rendered in
    three formats so all variants are evaluated on identical entity paths.

    Strategy:
      1. Pool all instances from CoT3 rule files
      2. Stratified split on answer (yes/no) -> shared train keys + test keys
      3. Render test instances in three formats:
           - Baseline : path facts only          (no symbolic tags)
           - CoT2     : path facts + POSITIVE/NEGATIVE
           - CoT3     : path facts + VALID/INVALID + POSITIVE/NEGATIVE
      4. train_keys returned so prepare_data can exclude test instances from training

    Returns:
        train_keys : set of (rule_id, instance_text) for training instances
        test_keys  : set of (rule_id, instance_text) for test instances
        test_baseline_df, test_cot2_df, test_cot3_df : DataFrames for evaluation
    """
    rule_context_cot3 = create_rule_context(rules_cot3)
    rule_context_cot2 = create_rule_context(rules_cot2)

    # Pool all CoT3 instances
    all_rows = []
    for rule in rules_cot3:
        for inst in rule["instances"]:
            all_rows.append({
                "_rule_id": rule["rule_id"],
                "_inst_key": inst["instance_text"],
                "_answer": inst.get("answer"),
                "_rule": rule,
                "_inst": inst,
            })

    if not all_rows:
        raise ValueError("No CoT3 instances found — cannot build shared test set.")

    all_df = pd.DataFrame(all_rows)

    # Warn and fill if answer tag missing (old-format rule files)
    missing = all_df["_answer"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} instances missing 'Answer: yes/no' tag. "
              f"Re-run NL-instances scripts to regenerate CoT files.")
        all_df["_answer"] = all_df["_answer"].fillna("yes")

    # Stratified split on answer (yes/no)
    train_parts, test_parts = [], []
    for key, group in all_df.groupby("_answer"):
        group = group.sample(frac=1).reset_index(drop=True)
        n_train = max(1, round(len(group) * train_ratio))
        if n_train == len(group) and len(group) >= 2:
            n_train = len(group) - 1
        train_parts.append(group.iloc[:n_train])
        test_parts.append(group.iloc[n_train:])

    train_pool = pd.concat(train_parts, ignore_index=True)
    test_pool  = pd.concat(test_parts,  ignore_index=True)

    test_keys  = set(zip(test_pool["_rule_id"],  test_pool["_inst_key"]))
    train_keys = set(zip(train_pool["_rule_id"], train_pool["_inst_key"]))

    print(f"\n  Shared test pool:  {len(test_pool)} instances "
          f"(yes={(test_pool['_answer']=='yes').sum()}, "
          f"no={(test_pool['_answer']=='no').sum()})")
    print(f"  Shared train pool: {len(train_pool)} instances")

    # --- Balance test pool if imbalanced ---
    # An imbalanced test set lets the model achieve high accuracy by always
    # predicting the majority class, collapsing Precision/Recall/F1 to 0.
    # We downsample the majority class (preferred over oversampling for test
    # sets — avoids duplicate instances inflating confidence).
    is_imbalanced, ratio, n_yes, n_no = _check_and_report_imbalance(
        test_pool, label="shared test pool", imbalance_threshold=imbalance_threshold
    )
    if is_imbalanced:
        minority_n = min(n_yes, n_no)
        yes_pool = test_pool[test_pool["_answer"] == "yes"].sample(
            n=minority_n, random_state=42
        )
        no_pool = test_pool[test_pool["_answer"] == "no"].sample(
            n=minority_n, random_state=42
        )
        test_pool = pd.concat([yes_pool, no_pool], ignore_index=True)\
            .sample(frac=1, random_state=42).reset_index(drop=True)
        # Update test_keys after downsampling
        test_keys = set(zip(test_pool["_rule_id"], test_pool["_inst_key"]))
        print(f"  Test pool balanced by downsampling majority: "
              f"{len(test_pool)} instances (yes={minority_n}, no={minority_n})")

    # Render test set in three formats using _build_cot_sample
    def _render_test(pool, fmt, rule_ctx):
        samples = []
        for _, row in pool.iterrows():
            inst = row["_inst"]
            rule = row["_rule"]
            sample = _build_cot_sample(
                inst, rule, rule_ctx, cot_format=fmt,
                include_reasoning=include_reasoning,
            )
            samples.append(sample)
        df = pd.DataFrame(samples)
        internal_cols = [c for c in df.columns if c.startswith("_")]
        return df.drop(columns=internal_cols)

    def _render_eval_test(pool):
        """
        Render the shared cross-format evaluation set.

        Every row contains:
          - Prompt  : path facts + yes/no question, NO rule context, NO tags
          - input_text : same path facts + question (for reference)
          - Label   : int 0/1 ground truth (never part of the prompt)

        This single file is used to evaluate ALL three models (Baseline,
        CoT2, CoT3) under identical conditions so their scores are directly
        comparable.
        """
        samples = []
        for _, row in pool.iterrows():
            inst = row["_inst"]
            rule = row["_rule"]

            # Re-use _build_cot_sample with Baseline format to get the
            # cleaned path_text and question — then build the eval-only prompt
            # separately so it never contains ###Response:\n<answer>.
            base_sample = _build_cot_sample(
                inst, rule, rule_context="", cot_format="Baseline",
                include_reasoning=False,
            )

            # path_text is everything in input_text before the question.
            # Recover it by stripping the question suffix from input_text.
            input_text = base_sample["input_text"]          # path_text + question
            question_part = input_text.split("Based on the rule")[0]  # everything before question
            # Safer: recompute path_text directly from cleaned instance
            path_text = _clean_instance_text(
                inst["instance_text"], cot_format="Baseline"
            ) + " "

            # Reconstruct question (same logic as _build_cot_sample)
            head_str = rule.get("head", "")
            head_match = re.match(
                r'(\?\w+)\s+(\S+)\s+(\S+)', head_str.strip()
            ) if head_str else None
            if head_match:
                head_pred = head_match.group(2)
                question = (
                    f"Based on the rule \"{rule.get('rule_text', '')}\" "
                    f"and the entailment above, "
                    f"is this {head_pred} relationship supported?"
                )
            else:
                question = (
                    f"Based on the rule \"{rule.get('rule_text', '')}\" "
                    f"and the entailment above, "
                    f"is this relationship supported?"
                )

            eval_prompt = _build_eval_only_prompt(path_text, question)

            samples.append({
                "Prompt":     eval_prompt,
                "input_text": path_text + question,
                "Label":      base_sample["Label"],
            })
        return pd.DataFrame(samples)

    test_baseline_df = _render_test(test_pool, "Baseline", rule_context_cot3)
    test_cot2_df     = _render_test(test_pool, "CoT2",     rule_context_cot2)
    test_cot3_df     = _render_test(test_pool, "CoT3",     rule_context_cot3)
    test_eval_df     = _render_eval_test(test_pool)

    return train_keys, test_keys, test_baseline_df, test_cot2_df, test_cot3_df, test_eval_df

def create_rule_context(rules: list, max_rules: int = 3) -> str:
    """Format rules into a context string for prompts.

    Only includes the rule text and PCA confidence value — the
    classification label is deliberately omitted to avoid leaking
    the answer into the input.
    """
    if not rules:
        return ""
    rule_context = "\n###Symbolic Rules (for reference):\n"
    for i, rule in enumerate(rules[:max_rules], 1):
        rule_context += f"{i}. {rule['rule_text']}\n"
        if rule["pca_confidence"] is not None:
            rule_context += (
                f"   (PCA Confidence: {rule['pca_confidence']:.3f})\n"
            )
    rule_context += "\n"
    return rule_context


# ---------------------------------------------------------------------------
# Training Data Generation — From Pre-generated CoT Rule Files (WITH rules)
# ---------------------------------------------------------------------------

def generate_training_data_with_rules(
    graph,
    node_list,
    relation2id,
    rules,
    max_path_length=10,
    include_reasoning=True,
    use_rules=True,
    max_rules_in_context=3,
    pca_threshold=0.5,
    train_ratio=0.8,
    total_samples=None,
    negative_mining=None,
    strip_shacl_from_input=False,
    excluded_keys=None,
    imbalance_threshold=1.5,
):
    """
    Generate training data from pre-computed CoT rule instances.

    excluded_keys: optional set of (rule_id, instance_text) tuples belonging
                   to the shared test set — these are filtered out before
                   building training samples to prevent data leakage.

    When use_rules=True:
        1. Pools ALL CoT instances from rule files
        2. Stratifies on classification (POSITIVE / NEGATIVE) for both
           CoT2 and CoT3, preserving the natural VALID/INVALID
           proportions in CoT3 (no validity balancing)
        3. Within each stratum, shuffles and splits 80:20 (train_ratio)
        4. Balances POSITIVE/NEGATIVE by oversampling the minority class
        Returns (train_df, test_df)

    When use_rules=False:
        Falls back to random-walk generation (no rule entailment).
        Uses total_samples to control output size.
        Returns (train_df, test_df) via random 80:20 split.
    """
    if not use_rules:
        all_df = _generate_random_walk_data(
            graph, node_list, relation2id,
            total_samples=total_samples or 1000,
            max_path_length=max_path_length,
            include_reasoning=include_reasoning,
        )
        # Stratified split on is_connected for random walk baseline
        train_df, test_df = _stratified_split(
            all_df, strat_col="is_connected", train_ratio=train_ratio,
        )
        return train_df, test_df

    # ----- Build from pre-generated CoT instances -----
    rule_context = create_rule_context(rules, max_rules=max_rules_in_context)

    # Detect CoT format from the first rule that has instances
    cot_format = "CoT2"
    for rule in rules:
        if rule["instances"]:
            cot_format = rule["cot_format"]
            break
    print(f"  Detected CoT format: {cot_format}")

    # Collect ALL instances and build samples (excluding shared test instances)
    all_samples = []
    n_excluded = 0
    for rule in rules:
        for inst in rule["instances"]:
            # Skip instances that belong to the shared test set
            if excluded_keys is not None:
                key = (rule["rule_id"], inst["instance_text"])
                if key in excluded_keys:
                    n_excluded += 1
                    continue
            sample = _build_cot_sample(
                inst, rule, rule_context, cot_format,
                include_reasoning=include_reasoning,
                strip_shacl_from_input=strip_shacl_from_input,
            )
            all_samples.append(sample)

    if n_excluded > 0:
        print(f"  Excluded {n_excluded} shared test instances from training pool.")

    if not all_samples:
        print("  WARNING: No CoT instances found. Falling back to random walk.")
        all_df = _generate_random_walk_data(
            graph, node_list, relation2id,
            total_samples=total_samples or 1000,
            max_path_length=max_path_length,
            include_reasoning=include_reasoning,
        )
        train_df, test_df = _stratified_split(
            all_df, strat_col="is_connected", train_ratio=train_ratio,
        )
        return train_df, test_df

    all_df = pd.DataFrame(all_samples)

    # --- Print pool statistics ---
    n_pos = (all_df["_classification"] == "POSITIVE").sum()
    n_neg = (all_df["_classification"] == "NEGATIVE").sum()
    n_yes = (all_df["_answer"] == "yes").sum()
    n_no  = (all_df["_answer"] == "no").sum()
    print(f"  Total CoT instances pooled: {len(all_df)}")
    print(f"    POSITIVE: {n_pos}  |  NEGATIVE: {n_neg}  (rule reliability signal)")
    print(f"    yes:      {n_yes}  |  no:       {n_no}    (ground truth answer)")

    if cot_format == "CoT3" and "_validity" in all_df.columns:
        print(f"    VALID:    {(all_df['_validity'] == 'VALID').sum()}")
        print(f"    INVALID:  {(all_df['_validity'] == 'INVALID').sum()}")

    # --- Stratification key: ground truth answer (yes/no) ---
    # This ensures both train and test have balanced yes/no distributions
    all_df["_strat_key"] = all_df["_answer"]

    # --- Stratified split ---
    train_df, test_df = _stratified_split(
        all_df, strat_col="_strat_key", train_ratio=train_ratio,
    )

    # --- Balance classes via oversampling if imbalanced ---
    train_df = _balance_classes(train_df, cot_format, imbalance_threshold)
    test_df  = _balance_classes(test_df,  cot_format, imbalance_threshold)

    # --- Final shuffle ---
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    # --- Report ---
    _print_split_report("Train", train_df, cot_format)
    _print_split_report("Test", test_df, cot_format)

    # --- Verify no leakage ---
    train_prompts = set(train_df["Prompt"])
    test_prompts = set(test_df["Prompt"])
    overlap = train_prompts & test_prompts
    if overlap:
        print(f"  WARNING: {len(overlap)} prompts appear in BOTH train and test!")
    else:
        print(f"  ✓ No prompt overlap between train and test sets.")

    # --- Compute sample weights for WeightedRandomSampler (Method 4) ---
    # Weights are attached to train_df only; test_df is always unweighted.
    # Logic:
    #   - Base weight for all samples = base_weight (default 1.0)
    #   - If enable_weighting=True AND CoT3 INVALID signal is present,
    #     NEGATIVE+INVALID samples get boosted to max_weight to emphasise
    #     hard-to-learn constraint violations.
    nm = negative_mining or {}
    if nm.get("enable", False):
        base_w  = float(nm.get("base_weight", 1.0))
        max_w   = float(nm.get("max_weight", base_w))
        do_shacl = nm.get("enable_weighting", False)

        train_df["weight"] = base_w

        if do_shacl and cot_format == "CoT3" and "_validity" in train_df.columns:
            # Boost NEGATIVE rows that also carry an INVALID constraint signal
            mask = (
                (train_df["_classification"] == "NEGATIVE") &
                (train_df["_validity"] == "INVALID")
            )
            train_df.loc[mask, "weight"] = max_w
            n_boosted = mask.sum()
            print(f"  [Weighting] base={base_w}, max={max_w}, "
                  f"boosted {n_boosted} NEGATIVE+INVALID rows")
        else:
            print(f"  [Weighting] Uniform weight={base_w} "
                  f"(SHACL boost {'disabled' if not do_shacl else 'N/A for ' + cot_format})")
    # If negative_mining is None or enable=False → no weight column → sampler skipped

    # --- Drop all internal/metadata columns — export only Prompt, input_text, output_text ---
    internal_cols = [c for c in train_df.columns if c.startswith("_")]
    train_df = train_df.drop(columns=internal_cols)
    test_df = test_df.drop(columns=internal_cols)

    return train_df, test_df


def _stratified_split(df, strat_col, train_ratio=0.8):
    """
    Split a DataFrame into train/test with proportional representation
    of each stratum.

    Each unique value of strat_col gets an 80:20 split (rounded),
    guaranteeing that even rare strata appear in both sets when possible.
    """
    train_parts = []
    test_parts = []

    for key, group in df.groupby(strat_col):
        group = group.sample(frac=1).reset_index(drop=True)  # shuffle
        n_train = max(1, round(len(group) * train_ratio))
        # Ensure at least 1 in test if group has ≥ 2 samples
        if n_train == len(group) and len(group) >= 2:
            n_train = len(group) - 1
        train_parts.append(group.iloc[:n_train])
        test_parts.append(group.iloc[n_train:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def _check_and_report_imbalance(df, label="dataset", imbalance_threshold=1.5):
    """
    Check yes/no imbalance in a DataFrame.

    Returns:
        (is_imbalanced, ratio, n_yes, n_no)
        is_imbalanced: True if majority/minority ratio exceeds imbalance_threshold
    """
    if "_answer" not in df.columns or len(df) == 0:
        return False, 1.0, 0, 0

    n_yes = (df["_answer"] == "yes").sum()
    n_no  = (df["_answer"] == "no").sum()

    if n_yes == 0 or n_no == 0:
        print(f"  WARNING [{label}]: Only one class present — yes={n_yes}, no={n_no}")
        return True, float("inf"), n_yes, n_no

    majority = max(n_yes, n_no)
    minority = min(n_yes, n_no)
    ratio = majority / minority

    is_imbalanced = ratio > imbalance_threshold
    status = f"IMBALANCED (ratio={ratio:.2f} > threshold={imbalance_threshold})" \
             if is_imbalanced else f"balanced (ratio={ratio:.2f})"
    print(f"  [{label}] yes={n_yes}, no={n_no} → {status}")

    return is_imbalanced, ratio, n_yes, n_no


def _balance_classes(df, cot_format, imbalance_threshold=1.5):
    """
    Balance the ground truth answer labels (yes/no) by oversampling
    the minority class — but ONLY if the imbalance ratio exceeds the threshold.

    imbalance_threshold: minimum majority/minority ratio to trigger balancing.
                         Default 1.5 means balancing kicks in when one class
                         is more than 1.5x the other.
    """
    if "_answer" not in df.columns or len(df) == 0:
        return df

    is_imbalanced, ratio, n_yes, n_no = _check_and_report_imbalance(
        df, imbalance_threshold=imbalance_threshold
    )

    if not is_imbalanced:
        return df  # Already balanced — no oversampling needed

    yes_df = df[df["_answer"] == "yes"]
    no_df  = df[df["_answer"] == "no"]
    target_per_class = max(len(yes_df), len(no_df))

    yes_balanced = _oversample_to(yes_df, target_per_class)
    no_balanced  = _oversample_to(no_df,  target_per_class)

    balanced = pd.concat([yes_balanced, no_balanced], ignore_index=True)
    print(f"  Balanced: yes={len(yes_balanced)}, no={len(no_balanced)} "
          f"(was yes={n_yes}, no={n_no})")
    return balanced


def _oversample_to(df, target_n):
    """Oversample a DataFrame to exactly target_n rows via repetition."""
    if len(df) == 0 or len(df) >= target_n:
        return df.copy()
    repeats = target_n // len(df)
    remainder = target_n % len(df)
    parts = [df] * repeats + [df.sample(n=remainder)]
    return pd.concat(parts, ignore_index=True)


def _print_split_report(label, df, cot_format):
    """Print distribution summary for a split."""
    print(f"\n  {label} set: {len(df)} samples")
    if "_answer" in df.columns:
        ans_counts = df["_answer"].value_counts()
        print(f"    Answer (yes/no):       {dict(ans_counts)}")
    if "_classification" in df.columns:
        cls_counts = df["_classification"].value_counts()
        print(f"    Classification (P/N):  {dict(cls_counts)}")
    if cot_format == "CoT3" and "_validity" in df.columns:
        val_counts = df["_validity"].value_counts()
        print(f"    Validity:       {dict(val_counts)}")
        # Cross-tab classification × validity
        ct = pd.crosstab(df["_classification"], df["_validity"])
        print(f"    Cross-tab (classification × validity):")
        for cls_val in ct.index:
            row = ", ".join(
                f"{v}={ct.loc[cls_val, v]}" for v in ct.columns
            )
            print(f"      {cls_val}: {row}")
        # Per-shape breakdown (v5 only — present when _shape_name is populated)
        if "_shape_name" in df.columns:
            invalid_df = df[df["_validity"] == "INVALID"]
            if not invalid_df.empty:
                shape_counts = invalid_df["_shape_name"].value_counts(dropna=False)
                print(f"    INVALID by shape:")
                for shape, count in shape_counts.items():
                    print(f"      {shape if shape else '(binary/unknown)'}: {count}")


# ---------------------------------------------------------------------------
# Sample Building (unchanged logic)
# ---------------------------------------------------------------------------

def _clean_instance_text(instance_text: str, cot_format: str,
                          strip_shacl_from_input: bool = False) -> str:
    """
    Clean the instance text for use as model input.

    Always removes:
      - The full classification sentence: "The path is classified as
        POSITIVE/NEGATIVE (PCA Confidence X.XXXX >= threshold 0.5)"
        including any dangling numeric fragments left by partial matches
      - The ground truth "Answer: yes/no" tag

    CoT3 (default): keeps [VALID]/[INVALID] bracket tags, normalises rich v5 tags
    CoT3 (strip_shacl=True): removes SHACL tags entirely (ablation mode)
    Baseline: strips all symbolic context, leaving only path facts
    """
    cleaned = instance_text

    # --- Step 1: Remove the entire classification + PCA sentence ---
    # Handles all variants:
    #   "The path is classified as POSITIVE (PCA Confidence 0.9982 >= threshold 0.5)"
    #   "The path is classified as NEGATIVE (PCA Confidence 0.3620 < threshold 0.5)"
    #   ". [VALID] The path is classified as POSITIVE (PCA Confidence 0.9273 >= threshold 0.5)"
    cleaned = re.sub(
        r'The path is classified as (?:POSITIVE|NEGATIVE)\s*'
        r'\(?(?:\s*PCA Confidence\s*[\d.]+\s*[<>=]+\s*threshold\s*[\d.]+\s*)?\)?',
        '', cleaned
    )

    # --- Step 2: Remove any remaining PCA confidence fragments ---
    # Catches orphaned fragments like "0.3620 < threshold 0.5)" or
    # "(PCA Confidence 0.9982 >= threshold 0.5)" that appear standalone
    cleaned = re.sub(
        r'\(?\s*(?:PCA Confidence\s*)?[\d.]+\s*[<>=]+\s*threshold\s*[\d.]+\s*\)?',
        '', cleaned
    )

    # --- Step 3: Handle SHACL tags based on format ---
    if cot_format == "Baseline":
        # Strip ALL symbolic context — only path facts remain
        cleaned = re.sub(r'\[(?:VALID|INVALID)[^\]]*\]', '', cleaned)
    elif strip_shacl_from_input and cot_format == "CoT3":
        # Remove ALL SHACL tags (ablation mode)
        cleaned = re.sub(r'\[(?:VALID|INVALID)[^\]]*\]', '', cleaned)
    else:
        # Normalise v5 rich tags → compact bracket form
        cleaned = re.sub(r'\[INVALID:[^\]]+\]', '[INVALID]', cleaned)
        cleaned = re.sub(r'\[VALID:[^\]]+\]', '[VALID]', cleaned)

    # --- Step 4: Remove ground truth answer tag ---
    cleaned = re.sub(r'\s*Answer:\s*(yes|no)\s*', ' ', cleaned, flags=re.IGNORECASE)

    # --- Step 5: Clean up punctuation and whitespace ---
    cleaned = re.sub(r'\.\s*\.', '.', cleaned)   # double dots
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'\s+([,.])', r'\1', cleaned)  # space before punctuation
    return cleaned


def _build_cot_sample(inst, rule, rule_context, cot_format,
                      include_reasoning=True,
                      strip_shacl_from_input=False):
    """
    Build a single training sample from a pre-generated CoT instance.

    The input contains only the cleaned entity path facts (tags and PCA
    comparison stripped by _clean_instance_text).  The output/reasoning
    contains the structured symbolic chain:
      - rule text + PCA confidence
      - CoT3 v4: "The entailed path is VALID/INVALID."
      - CoT3 v5: "The entailed path violates <ShapeName> (<description>)."
                 or "The entailed path satisfies all SHACL constraints."
      - final POSITIVE/NEGATIVE + yes/no label

    Output CSV columns: Prompt, input_text, output_text (no metadata).
    """
    instance_text = inst["instance_text"]
    classification = inst["classification"]
    validity = inst.get("validity")
    shape_name = inst.get("shape_name")
    shape_description = inst.get("shape_description")
    pca_confidence = inst.get("pca_confidence")
    rule_text = rule.get("rule_text", "")

    # Label: parse ground truth from "Answer: yes/no" embedded in instance text
    answer_match = re.search(r'Answer:\s*(yes|no)', instance_text, re.IGNORECASE)
    if answer_match:
        label = answer_match.group(1).lower()
    else:
        # Fallback for old-format rule files without embedded answer
        # (POSITIVE/NEGATIVE is rule reliability, not ground truth — log a warning)
        import warnings
        warnings.warn(
            f"No 'Answer: yes/no' found in instance text. "
            f"Falling back to POSITIVE→yes / NEGATIVE→no which may be incorrect. "
            f"Please regenerate CoT files with updated NL-instances scripts.",
            UserWarning, stacklevel=2
        )
        label = "yes" if classification == "POSITIVE" else "no"

    # Clean the instance text: strip classification/PCA leaks, keep path facts
    path_text = _clean_instance_text(
        instance_text, cot_format,
        strip_shacl_from_input=strip_shacl_from_input
    ) + " "

    # Build question based on the rule's head predicate
    head_str = rule.get("head", "")
    head_match = re.match(
        r'(\?\w+)\s+(\S+)\s+(\S+)', head_str.strip()
    ) if head_str else None

    if head_match:
        head_pred = head_match.group(2)
        question = (
            f"Based on the rule \"{rule_text}\" and the entailment above, "
            f"is this {head_pred} relationship supported?"
        )
    else:
        question = (
            f"Based on the rule \"{rule_text}\" and the entailment above, "
            f"is this relationship supported?"
        )

    # Build answer — model derives all conclusions here
    if include_reasoning:
        reasoning = f"The rule states: \"{rule_text}\". "
        if pca_confidence is not None:
            reasoning += (
                f"The PCA Confidence of this rule is {pca_confidence:.4f}. "
            )
        if cot_format == "CoT3" and validity is not None:
            # Keep reasoning consistent with the [VALID]/[INVALID] tags in the input
            if validity == "INVALID":
                reasoning += "The entailed path is INVALID based on SHACL Constraints. "
            else:
                reasoning += "The entailed path is VALID based on SHACL Constraints. "
        reasoning += (
            f"The path is classified as {classification}. "
            f"The answer is {label}."
        )
        answer = reasoning
    else:
        answer = f"The answer is {label}."

    # Build full prompt
    prompt = _build_prompt(
        path_text, question, answer, rule_context,
        use_rules=True, include_reasoning=include_reasoning,
    )

    # Return columns needed for training.
    # Label (int 0/1) is kept as an explicit exported column so evaluate_model()
    # always has a reliable ground truth regardless of output_text parsing.
    # It does NOT start with "_" so it survives the internal-column purge.
    sample = {
        "Prompt": prompt,
        "input_text": path_text + question,
        "output_text": answer,
        "Label": 1 if label == "yes" else 0,
    }

    # Keep classification and answer internally for stratified splitting (dropped before CSV export)
    sample["_classification"] = classification
    sample["_answer"] = label   # ground truth yes/no — used for stratification and balancing
    if cot_format == "CoT3":
        sample["_validity"] = validity
        sample["_shape_name"] = shape_name  # kept for diagnostics, dropped before export

    return sample


def _build_eval_only_prompt(path_text: str, question: str) -> str:
    """
    Build the shared cross-format evaluation prompt.

    Contains ONLY:
      - The ###Instruction header (identical for all three variants)
      - path facts (entity names, relations — no [VALID]/[INVALID] tags,
        no POSITIVE/NEGATIVE, no PCA confidence, no rule context)
      - the yes/no question
      - the ###Response: marker (model generates from here)

    This is the single prompt format used to evaluate ALL three fine-tuned
    models (Baseline, CoT2, CoT3) fairly — none of them see any symbolic
    signal at inference time.

    The answer field is intentionally omitted so that when this prompt is
    stored in the CSV the model never sees the ground truth.  The Label
    column carries the ground truth for the evaluation script.
    """
    return (
        f"###Instruction:\nAnswer the following yes/no question by "
        f"reasoning step-by-step.\n\n"
        f"###Input:\n{path_text}{question}\n\n"
        f"###Response:"
    )


def _build_prompt(path_text, question, answer, rule_context,
                  use_rules=True, include_reasoning=True):
    """Build the formatted prompt string."""
    if include_reasoning:
        if use_rules and rule_context:
            prompt = (
                f"###Instruction:\nAnswer the following yes/no question by "
                f"reasoning step-by-step. Use the symbolic rules as additional "
                f"context along with the path information.\n{rule_context}"
                f"###Input:\n{path_text}{question}\n\n###Response:\n{answer}"
            )
        else:
            prompt = (
                f"###Instruction:\nAnswer the following yes/no question by "
                f"reasoning step-by-step.\n\n###Input:\n{path_text}{question}"
                f"\n\n###Response:\n{answer}"
            )
    else:
        if use_rules and rule_context:
            prompt = (
                f"{rule_context}###Input:\n{path_text}{question}"
                f"\n\n###Response:\n{answer}"
            )
        else:
            prompt = (
                f"###Input:\n{path_text}{question}\n\n###Response:\n{answer}"
            )
    return prompt


# ---------------------------------------------------------------------------
# Training Data Generation — Random Walk (WITHOUT rules, baseline)
# ---------------------------------------------------------------------------

def _generate_random_walk_data(
    graph,
    node_list,
    relation2id,
    total_samples=1000,
    max_path_length=10,
    include_reasoning=True,
):
    """
    Generate training data via random walks (no rule entailment).
    This is the 'without rules' experiment baseline.
    Original logic preserved with safety limits.
    """
    data = []
    unique_paths = set()
    pos_count = 0
    neg_count = 0
    target_per_class = total_samples // 2

    max_outer_attempts = total_samples * 50
    outer_attempts = 0

    while len(data) < total_samples and outer_attempts < max_outer_attempts:
        outer_attempts += 1

        path_length = random.randint(2, max_path_length)
        first_node = random.choice(node_list)
        visited = {first_node}
        path_text = ""
        reasoning_text = ""
        previous_node = first_node

        for step in range(path_length - 1):
            if previous_node not in graph or not graph[previous_node]:
                node = random.choice(node_list)
                safety_counter = 0
                while node in visited and safety_counter < 100:
                    node = random.choice(node_list)
                    safety_counter += 1
                if node in visited:
                    break

                path_text += (
                    f"node_{previous_node} not connected with node_{node}. "
                )
                if include_reasoning:
                    reasoning_text += (
                        f"node_{previous_node} not connected with node_{node} "
                        f"means there is no relationship. "
                    )
                visited.add(node)
                previous_node = node
            else:
                next_node = random.choice(list(graph[previous_node].keys()))
                safety_counter = 0
                while (
                    next_node in visited
                    and len(visited) < len(node_list)
                    and safety_counter < 100
                ):
                    next_node = random.choice(
                        list(graph[previous_node].keys())
                    )
                    safety_counter += 1

                relation = graph[previous_node][next_node]
                rel_name = relation2id.get(relation, f"relation_{relation}")
                path_text += (
                    f"node_{previous_node} has {rel_name} "
                    f"with node_{next_node}. "
                )
                if include_reasoning:
                    reasoning_text += (
                        f"node_{previous_node} has {rel_name} "
                        f"with node_{next_node}. "
                    )
                visited.add(next_node)
                previous_node = next_node

        last_node = previous_node

        if not path_text or path_text in unique_paths:
            continue
        unique_paths.add(path_text)

        question = f"Is node_{first_node} connected with node_{last_node}?"
        is_connected = (
            first_node in graph and last_node in graph[first_node]
        ) or (last_node in graph and first_node in graph[last_node])

        if is_connected and pos_count >= target_per_class:
            continue
        if not is_connected and neg_count >= target_per_class:
            continue

        if is_connected:
            if include_reasoning:
                answer = reasoning_text + "The answer is yes."
            else:
                answer = "The answer is yes."
            pos_count += 1
        else:
            if include_reasoning:
                answer = reasoning_text + "The answer is no."
            else:
                answer = "The answer is no."
            neg_count += 1

        prompt = _build_prompt(
            path_text, question, answer, rule_context="",
            use_rules=False, include_reasoning=include_reasoning,
        )

        data.append({
            "Prompt": prompt,
            "input_text": path_text + question,
            "output_text": answer,
            "Label": 1 if is_connected else 0,
            "has_rule_context": False,
            "is_connected": is_connected,
        })

        if len(data) % 200 == 0:
            print(f"  Generated {len(data)}/{total_samples} samples...")

    if len(data) < total_samples:
        print(f"  WARNING: Only generated {len(data)}/{total_samples} samples")

    print(
        f"  Generated {len(data)} samples "
        f"(positive: {pos_count}, negative: {neg_count})"
    )
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Model Utilities (require GPU libraries)
# ---------------------------------------------------------------------------

def _check_ml_available():
    """Raise a clear error if ML libraries weren't loaded."""
    if not _ML_AVAILABLE:
        raise ImportError(
            "GPU/ML libraries (torch, transformers, peft, bitsandbytes) "
            "are not available. This is likely due to a missing system "
            "library (e.g. GLIBCXX_3.4.29). These are required for "
            "model training and evaluation but NOT for data preparation."
        )


def create_bnb_config(cfg: dict) -> "BitsAndBytesConfig":
    """Create BitsAndBytes quantization config from the JSON config."""
    _check_ml_available()
    q = cfg.get("quantization", {})
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    compute_dtype = dtype_map.get(
        q.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16
    )
    return BitsAndBytesConfig(
        load_in_4bit=q.get("load_in_4bit", True),
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model(model_name: str, bnb_config, max_memory_mb: int = 40960):
    """Load a quantized model and tokenizer.

    max_memory is specified for BOTH GPU and CPU to prevent device_map="auto"
    from silently offloading unlimited model layers to CPU RAM (which causes
    the OOM kill seen in SLURM jobs that die within 2 minutes of starting).

    CPU RAM cap: 24GB — enough headroom for the OS, data loaders, and
    tokenizer while preventing runaway offloading.
    """
    _check_ml_available()
    n_gpus = torch.cuda.device_count()
    print(f"  Detected {n_gpus} GPU(s). GPU memory cap: {max_memory_mb}MB each.")

    if n_gpus == 0:
        raise RuntimeError(
            "No GPUs detected by torch.cuda.device_count(). "
            "Check that the SLURM job has --gres=gpu:1 and that CUDA is "
            "accessible in the conda environment."
        )

    gpu_max = f"{max_memory_mb}MB"
    cpu_max = "24GiB"   # hard cap on CPU RAM offloading

    max_memory = {i: gpu_max for i in range(n_gpus)}
    max_memory["cpu"] = cpu_max

    print(f"  max_memory: GPU={gpu_max}, CPU={cpu_max}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
    )

    # Report where each layer landed
    if hasattr(model, "hf_device_map"):
        devices = {}
        for layer, dev in model.hf_device_map.items():
            devices[str(dev)] = devices.get(str(dev), 0) + 1
        print(f"  Layer placement: {devices}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def find_all_linear_names(model) -> list:
    """Find all Linear4bit module names for LoRA targeting."""
    _check_ml_available()
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(
                names[0] if len(names) == 1 else names[-1]
            )
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def preprocess_batch(batch, tokenizer, max_length=512):
    """Tokenize a batch of prompts."""
    return tokenizer(
        batch["Prompt"], truncation=True, max_length=max_length,
        padding="max_length",
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model, tokenizer, test_df, max_samples=200, max_new_tokens=50,
    device="cuda",
) -> dict:
    """
    Evaluate a model on the test set. Returns metrics dict.

    Ground truth resolution (in priority order):
      1. Label column (int 0/1) — always present in CSVs generated by this
         pipeline. This is the canonical, reliable source.
      2. Regex on output_text for "The answer is yes/no" — fallback for
         legacy CSVs that predate the Label column.

    If neither source is available for a row, it is skipped and a warning
    is printed so the caller can regenerate the data.

    Truncation guard: prompts longer than the tokenizer's model_max_length
    are truncated on the LEFT (keeping the ###Response: suffix) so the
    model always generates from the correct position. A count of truncated
    prompts is reported at the end.
    """
    _check_ml_available()
    model.eval()
    y_true = []
    y_pred = []

    n = min(len(test_df), max_samples)
    print(f"Evaluating on {n} samples...")

    # Check whether Label column is present and warn once if not
    has_label_col = "Label" in test_df.columns
    if not has_label_col:
        print(
            "  WARNING: 'Label' column not found in test CSV. "
            "Falling back to parsing output_text for ground truth. "
            "Regenerate CSVs with the updated pipeline to fix this."
        )

    n_fallback = 0       # rows where Label was absent → fell back to regex
    n_skipped  = 0       # rows where neither source gave a valid label
    n_truncated = 0      # prompts that exceeded max_length and were truncated

    # Resolve tokenizer max length once (guards against very long YAGO entity names)
    tok_max = getattr(tokenizer, "model_max_length", 2048)
    # Some tokenisers report absurdly large values (e.g. 1e30); cap sensibly
    if tok_max > 8192:
        tok_max = 2048

    for idx, row in test_df.head(n).iterrows():

        # ------------------------------------------------------------------
        # 1. Build eval prompt (everything up to and including ###Response:)
        # ------------------------------------------------------------------
        prompt = row["Prompt"]
        if "###Response:" in prompt:
            eval_prompt = prompt[:prompt.index("###Response:") + len("###Response:")]
        else:
            eval_prompt = prompt

        # ------------------------------------------------------------------
        # 2. Resolve ground truth
        # ------------------------------------------------------------------
        if has_label_col and pd.notna(row.get("Label")):
            expected_yes = int(row["Label"]) == 1
        else:
            # Fallback: parse "The answer is yes/no" from output_text
            gt_match = re.search(
                r'The answer is (yes|no)', str(row.get("output_text", "")),
                re.IGNORECASE
            )
            if gt_match:
                expected_yes = gt_match.group(1).lower() == "yes"
                n_fallback += 1
            else:
                # Cannot determine ground truth — skip this row
                n_skipped += 1
                continue

        # ------------------------------------------------------------------
        # 3. Tokenise with truncation guard
        #    Truncate from the LEFT so ###Response: is always at the end.
        # ------------------------------------------------------------------
        inputs = tokenizer(
            eval_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=tok_max,
        ).to(device)

        # Detect if truncation occurred (tokenised length == tok_max)
        if inputs["input_ids"].shape[1] == tok_max:
            n_truncated += 1

        input_len = inputs["input_ids"].shape[1]

        # ------------------------------------------------------------------
        # 4. Generate only new tokens
        # ------------------------------------------------------------------
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        # Decode ONLY newly generated tokens (avoids 'no' in entity names
        # inside the prompt being matched by the extraction regex)
        new_tokens = outputs[0][input_len:]
        model_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # ------------------------------------------------------------------
        # 5. Extract yes/no from model output
        #    Priority: "The answer is X" pattern → word-boundary match → last occurrence
        # ------------------------------------------------------------------
        ans_match = re.search(r'the answer is (yes|no)', model_answer.lower())
        if ans_match:
            model_has_yes = ans_match.group(1) == "yes"
        else:
            has_yes = bool(re.search(r'\byes\b', model_answer.lower()))
            has_no  = bool(re.search(r'\bno\b',  model_answer.lower()))
            if has_yes and not has_no:
                model_has_yes = True
            elif has_no and not has_yes:
                model_has_yes = False
            else:
                # Ambiguous or empty — use last occurrence as tiebreaker
                last_yes = model_answer.lower().rfind("yes")
                last_no  = model_answer.lower().rfind("no")
                model_has_yes = last_yes > last_no

        y_true.append(expected_yes)
        y_pred.append(model_has_yes)

        if (len(y_true)) % 50 == 0:
            print(f"  Processed {len(y_true)}/{n} samples...")

    # ------------------------------------------------------------------
    # 6. Diagnostics before computing metrics
    # ------------------------------------------------------------------
    n_eval = len(y_true)
    pred_yes = sum(y_pred)
    pred_no  = n_eval - pred_yes
    true_yes = sum(y_true)
    true_no  = n_eval - true_yes

    print(f"\n  Evaluation summary ({n_eval} samples evaluated):")
    print(f"    Ground truth  — yes: {true_yes}, no: {true_no}")
    print(f"    Predictions   — yes: {pred_yes}, no: {pred_no}")
    if n_fallback:
        print(f"    Fallback label resolution (no Label col): {n_fallback} rows")
    if n_skipped:
        print(f"    WARNING: Skipped {n_skipped} rows (no resolvable ground truth)")
    if n_truncated:
        print(
            f"    WARNING: {n_truncated}/{n_eval} prompts were truncated to "
            f"{tok_max} tokens. Consider increasing max_length in config.json."
        )

    if n_eval == 0:
        print("  ERROR: No samples could be evaluated. Check Label column and output_text format.")
        return {
            "accuracy": 0.0, "f1_score": 0.0,
            "precision": 0.0, "recall": 0.0,
            "y_true": [], "y_pred": [],
        }

    accuracy  = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, pos_label=True, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=True, zero_division=0)
    recall    = recall_score(y_true, y_pred, pos_label=True, zero_division=0)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def fine_tune_model(model, tokenizer, train_dataset, output_dir: str,
                    cfg: dict, sampler=None):
    """Fine-tune a model with LoRA + QLoRA. Returns the fine-tuned model.

    Args:
        sampler: Optional WeightedRandomSampler (Method 4). Injected via
                 _get_train_sampler so the standard HF DataLoader and collator
                 are completely untouched — only sampling order changes.
    """
    _check_ml_available()
    tcfg = cfg.get("training", {})
    lcfg = cfg.get("lora", {})

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    modules = find_all_linear_names(model)
    peft_config = LoraConfig(
        r=lcfg.get("r", 16),
        lora_alpha=lcfg.get("lora_alpha", 64),
        target_modules=modules,
        lora_dropout=lcfg.get("lora_dropout", 0.1),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)"
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 4),
        warmup_steps=tcfg.get("warmup_steps", 10),
        max_steps=tcfg.get("num_steps", 500),
        learning_rate=tcfg.get("learning_rate", 2e-4),
        fp16=tcfg.get("fp16", True),
        logging_steps=tcfg.get("logging_steps", 50),
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=tcfg.get("save_steps", 100),
    )

    if sampler is not None:
        _sampler = sampler

        class _WeightedTrainer(Trainer):
            # HF passes `dataset` as a positional arg in newer versions;
            # accept it but ignore it — we always return the fixed sampler.
            def _get_train_sampler(self, dataset=None):
                return _sampler

        print("WeightedRandomSampler active (Method 4)")
        trainer = _WeightedTrainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
    else:
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

    model.config.use_cache = False
    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    return model


def print_results(label: str, results: dict):
    """Pretty-print evaluation results."""
    print(f"\n{label}")
    print(f"   Accuracy:  {results['accuracy']:.3f}")
    print(f"   F1 Score:  {results['f1_score']:.3f}")
    print(f"   Precision: {results['precision']:.3f}")
    print(f"   Recall:    {results['recall']:.3f}")


def save_results_json(path: str, results: dict):
    """Save results dict to JSON."""
    clean = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean[k] = {
                kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                for kk, vv in v.items()
                if kk not in ("y_true", "y_pred")
            }
        elif isinstance(v, (float, np.floating)):
            clean[k] = float(v)
        else:
            clean[k] = v
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"Results saved to {path}")
