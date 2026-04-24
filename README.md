# NeSy-KG-LLM
=======
# NeSyKGLLM — Server Pipeline

Modular scripts for **Neuro-Symbolic Knowledge Graph Link Prediction with LLMs**,
converted from the Colab notebook for headless GPU server execution.

## Project Structure

```
nesykgllm/
-- config.json                  # ← Edit this with your paths and parameters
-- utils.py                     # Shared functions (KG loading, rules, training, eval)
-- prepare_data.py              # Preprocess KG + generate training/test CSVs
-- step1_evaluate_base.py       # Evaluate pre-trained model (no fine-tuning)
-- step2_finetune_no_rules.py   # Fine-tune WITHOUT symbolic rules
-- step3_finetune_with_rules.py # Fine-tune WITH symbolic rules
-- compare_results.py           # Aggregate results + produce comparison chart
-- README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install bitsandbytes transformers peft accelerate datasets \
            sentencepiece scikit-learn torch trl matplotlib pandas
```

### 2. Edit `config.json`

Set your HuggingFace token, data paths, and any hyperparameters you want to adjust.

Key fields to change:
- `huggingface_token`- your HF API token
- `data_dir` - directory containing `train2id.txt` and `relation2id.txt`
- `rules_dir` - directory containing `rule_*.txt` files
- `output_dir` - where models, CSVs, and results will be saved

### 3. Prepare data (run once)

```bash
python prepare_data.py --config config.json
```

This generates:
- `outputs/train_data_with_rules.csv`
- `outputs/train_data_without_rules.csv`
- `outputs/test_data.csv`

### 4. Run experiments

```bash
# Step 1: Evaluate base model
python step1_evaluate_base.py --config config.json

# Step 2: Fine-tune without rules
python step2_finetune_no_rules.py --config config.json

# Step 3: Fine-tune with rules
python step3_finetune_with_rules.py --config config.json
```

Each step saves its results JSON and model checkpoints into `output_dir`.

### 5. Compare results

```bash
python compare_results.py --config config.json
```

Produces `comprehensive_comparison_results.json` and `model_comparison.png`.

## Using Pre-generated CSVs

If you already have training/test CSVs (e.g., from a previous run or from YAGO3-10),
set in `config.json`:

```json
"pregenerated_data": {
    "use_pregenerated": true,
    "train_with_rules_csv": "/path/to/train_data_with_rules.csv",
    "train_without_rules_csv": "/path/to/train_data_without_rules.csv",
    "test_csv": "/path/to/test_data.csv"
}
```

Then skip `prepare_data.py` and run the step scripts directly.

## Output Structure

After all steps complete:

```
outputs/
- train_data_with_rules.csv
- train_data_without_rules.csv
- test_data.csv
- base_model_results.json
- finetune_no_rules_results.json
- finetune_with_rules_results.json
- comprehensive_comparison_results.json
- model_comparison.png
- finetuned_LLaMA_3_8B_baseline/    # LoRA adapter weights
- finetuned_LLaMA_3_8B_with_rules/  # LoRA adapter weights
```
