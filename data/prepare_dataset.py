"""
data/prepare_dataset.py
-----------------------
Downloads the instruction dataset from HuggingFace Hub, formats it into
prompt/response pairs, splits into train/eval, and saves locally as JSON.
"""

import json
import os
import random
import yaml
from pathlib import Path
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

DATASET_ID   = cfg["dataset"]["hf_dataset_id"]
SPLIT        = cfg["dataset"]["split"]
EVAL_RATIO       = cfg["dataset"]["eval_ratio"]
MAX_LENGTH       = cfg["dataset"]["max_length"]
MAX_TRAIN_SAMPLES = cfg["dataset"].get("max_train_samples", None)
NUM_WORKERS  = cfg["dataset"]["num_workers"]
OUT_DIR      = ROOT / "data"

SEED = 42
random.seed(SEED)

# ── Prompt template ───────────────────────────────────────────────────────────
def format_example(example: dict) -> dict | None:
    """
    Format a raw dataset row into a single 'text' field using an
    instruction-following template.

    Expected columns from iamtarun/python_code_instructions_18k_alpaca:
        instruction, input (optional), output
    """
    instruction = (example.get("instruction") or "").strip()
    inp         = (example.get("input")       or "").strip()
    output      = (example.get("output")      or "").strip()

    if not instruction or not output:
        return None

    if inp:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )

    return {"text": prompt}


# ── Load & format ─────────────────────────────────────────────────────────────
def main():
    print(f"Loading dataset: {DATASET_ID} ...")
    raw = load_dataset(DATASET_ID, split=SPLIT, num_proc=NUM_WORKERS if NUM_WORKERS > 0 else None)
    print(f"  Raw rows: {len(raw)}")

    formatted = []
    skipped   = 0
    for row in raw:
        result = format_example(row)
        if result is None:
            skipped += 1
            continue
        # Skip examples that are clearly too long (rough char estimate)
        if len(result["text"]) > MAX_LENGTH * 6:   # ~6 chars/token heuristic
            skipped += 1
            continue
        formatted.append(result)

    print(f"  Formatted: {len(formatted)}  |  Skipped: {skipped}")

    # Apply max_train_samples cap before split
    if MAX_TRAIN_SAMPLES is not None:
        total_cap = int(MAX_TRAIN_SAMPLES / (1 - EVAL_RATIO))  # account for eval split
        formatted = formatted[:total_cap]
        print(f"  Capped to: {len(formatted)} samples (max_train_samples={MAX_TRAIN_SAMPLES})")

    # ── Train / eval split ────────────────────────────────────────────────────
    random.shuffle(formatted)
    eval_size  = max(1, int(len(formatted) * EVAL_RATIO))
    eval_data  = formatted[:eval_size]
    train_data = formatted[eval_size:]

    print(f"  Train: {len(train_data)}  |  Eval: {len(eval_data)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.json"
    eval_path  = OUT_DIR / "eval.json"

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    print(f"\nSaved:")
    print(f"  {train_path}  ({train_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {eval_path}  ({eval_path.stat().st_size / 1e6:.1f} MB)")

    # ── Sanity check — print 1 example ───────────────────────────────────────
    print("\n── Sample training example ──────────────────────────────────")
    print(train_data[0]["text"][:600])
    print("...")


if __name__ == "__main__":
    main()