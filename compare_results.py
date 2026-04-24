#!/usr/bin/env python3
"""
NeSyKGLLM - Compare Results
=============================
Loads the JSON results from all three steps and produces a comparison
summary + visualization chart.

Usage:
    python compare_results.py --config config.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from utils import load_config


def load_step_results(path: str) -> dict:
    """Load a step results JSON, returning the metrics sub-dict."""
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("metrics", data)


def main():
    parser = argparse.ArgumentParser(description="NeSyKGLLM - Compare Results")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg["output_dir"]

    # ------------------------------------------------------------------
    # Load results
    # ------------------------------------------------------------------
    base = load_step_results(os.path.join(output_dir, "base_model_results.json"))
    no_rules = load_step_results(os.path.join(output_dir, "finetune_no_rules_results.json"))
    with_rules = load_step_results(os.path.join(output_dir, "finetune_with_rules_results.json"))

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("=" * 70)
    print("FINAL COMPARISON RESULTS")
    print("=" * 70)

    comparison_df = pd.DataFrame({
        "Model": ["Base Model", "Fine-tuned WITHOUT Rules", "Fine-tuned WITH Rules"],
        "Accuracy": [base["accuracy"], no_rules["accuracy"], with_rules["accuracy"]],
        "F1 Score": [base["f1_score"], no_rules["f1_score"], with_rules["f1_score"]],
        "Precision": [base["precision"], no_rules["precision"], with_rules["precision"]],
        "Recall": [base["recall"], no_rules["recall"], with_rules["recall"]],
    })
    print(comparison_df.to_string(index=False))

    # Improvements
    bl_acc = (no_rules["accuracy"] - base["accuracy"]) * 100
    ru_acc = (with_rules["accuracy"] - base["accuracy"]) * 100
    rv_acc = (with_rules["accuracy"] - no_rules["accuracy"]) * 100
    bl_f1 = (no_rules["f1_score"] - base["f1_score"]) * 100
    ru_f1 = (with_rules["f1_score"] - base["f1_score"]) * 100
    rv_f1 = (with_rules["f1_score"] - no_rules["f1_score"]) * 100

    print(f"\nACCURACY IMPROVEMENTS:")
    print(f"   Fine-tuning (no rules) vs Base:   {bl_acc:+.1f} pp")
    print(f"   Fine-tuning (with rules) vs Base:  {ru_acc:+.1f} pp")
    print(f"   With rules vs Without rules:       {rv_acc:+.1f} pp")
    print(f"\nF1 SCORE IMPROVEMENTS:")
    print(f"   Fine-tuning (no rules) vs Base:   {bl_f1:+.1f} pp")
    print(f"   Fine-tuning (with rules) vs Base:  {ru_f1:+.1f} pp")
    print(f"   With rules vs Without rules:       {rv_f1:+.1f} pp")

    # ------------------------------------------------------------------
    # Save comprehensive JSON
    # ------------------------------------------------------------------
    summary = {
        "model": cfg["model_key"],
        "base_model": base,
        "finetuned_without_rules": no_rules,
        "finetuned_with_rules": with_rules,
        "improvements": {
            "baseline_vs_base_accuracy": bl_acc,
            "rules_vs_base_accuracy": ru_acc,
            "rules_vs_baseline_accuracy": rv_acc,
            "baseline_vs_base_f1": bl_f1,
            "rules_vs_base_f1": ru_f1,
            "rules_vs_baseline_f1": rv_f1,
        },
    }

    summary_path = os.path.join(output_dir, "comprehensive_comparison_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nComprehensive results saved to {summary_path}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    models = ["Base\nModel", "Fine-tuned\n(No Rules)", "Fine-tuned\n(With Rules)"]
    accuracies = [base["accuracy"], no_rules["accuracy"], with_rules["accuracy"]]
    f1_scores = [base["f1_score"], no_rules["f1_score"], with_rules["f1_score"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1"]

    bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.8)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Accuracy Comparison", fontsize=14, fontweight="bold")
    ax1.set_ylim([0, 1])
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    bars2 = ax2.bar(models, f1_scores, color=colors, alpha=0.8)
    ax2.set_ylabel("F1 Score", fontsize=12)
    ax2.set_title("F1 Score Comparison", fontsize=14, fontweight="bold")
    ax2.set_ylim([0, 1])
    ax2.grid(axis="y", alpha=0.3)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {chart_path}")


if __name__ == "__main__":
    main()
