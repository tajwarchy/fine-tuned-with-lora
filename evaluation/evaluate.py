"""
evaluation/evaluate.py
----------------------
1. Computes perplexity on the held-out eval set for:
   - Base model (TinyLlama, no fine-tuning)
   - Fine-tuned merged model
2. Runs 10 domain-specific prompts through both models side by side
3. Saves results to evaluation/results/
4. Logs perplexity scores to MLflow
"""

import json
import math
import time
import yaml
import torch
import mlflow
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_ID      = cfg["model"]["base_model_id"]
MERGED_DIR    = ROOT / cfg["paths"]["merged_model_dir"]
EVAL_JSON     = ROOT / "data" / "eval.json"
RESULTS_DIR   = ROOT / cfg["paths"]["eval_results_dir"]
MLFLOW_CFG    = cfg["mlflow"]
INFER_CFG     = cfg["inference"]
MAX_LENGTH    = cfg["dataset"]["max_length"]

DEVICE = torch.device("cpu")   # MPS lacks some ops (aten::isin) — use CPU for eval
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Perplexity dataset ────────────────────────────────────────────────────────
class EvalDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        with open(path) as f:
            self.examples = json.load(f)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.examples[idx]["text"],
            truncation    = True,
            max_length    = self.max_length,
            padding       = "max_length",
            return_tensors= "pt",
        )
        input_ids      = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        labels         = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ── Perplexity ────────────────────────────────────────────────────────────────
def compute_perplexity(model, tokenizer, limit=20) -> float:
    """Compute perplexity on up to `limit` eval examples."""
    dataset    = EvalDataset(EVAL_JSON, tokenizer, MAX_LENGTH)
    dataset.examples = dataset.examples[:limit]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"\r  Perplexity eval: {i+1}/{len(dataloader)}", end="", flush=True)
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss  += outputs.loss.item()
            total_steps += 1

    print()
    avg_loss   = total_loss / total_steps
    perplexity = math.exp(avg_loss)
    return perplexity


# ── Qualitative prompts ───────────────────────────────────────────────────────
PROMPTS = [
    "### Instruction:\nWrite a Python function to check if a number is prime.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function to reverse a string.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function to find the factorial of a number using recursion.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function to merge two sorted lists.\n\n### Response:\n",
    "### Instruction:\nWrite a Python class for a stack data structure with push, pop, and peek methods.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function that takes a list of numbers and returns only the even ones.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function to count word frequency in a string.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function to flatten a nested list.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function to binary search a sorted list.\n\n### Response:\n",
    "### Instruction:\nWrite a Python decorator that measures the execution time of a function.\n\n### Response:\n",
]


def generate(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens    = cfg["inference"]["max_new_tokens"],
            temperature       = cfg["inference"]["temperature"],
            top_p             = cfg["inference"]["top_p"],
            repetition_penalty= cfg["inference"]["repetition_penalty"],
            do_sample         = True,
            pad_token_id      = tokenizer.eos_token_id,
        )
    # Return only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def load_model(model_path: str, label: str):
    print(f"\nLoading {label} from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype       = torch.float16,
        low_cpu_mem_usage = True,
    )
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CFG["experiment_name"])

    results = {"perplexity": {}, "qualitative": []}

    with mlflow.start_run(run_name="evaluation"):

        for label, model_path in [
            ("base",       MODEL_ID),
            ("fine-tuned", str(MERGED_DIR)),
        ]:
            print(f"\n{'═'*60}")
            print(f"  Evaluating: {label.upper()}")
            print(f"{'═'*60}")

            model, tokenizer = load_model(model_path, label)

            # ── Perplexity ────────────────────────────────────────────────
            print(f"\nComputing perplexity ({label}) ...")
            ppl = compute_perplexity(model, tokenizer)
            results["perplexity"][label] = ppl
            print(f"  Perplexity [{label}]: {ppl:.2f}")
            mlflow.log_metric(f"perplexity_{label.replace('-','_')}", ppl)

            # ── Qualitative ───────────────────────────────────────────────
            print(f"\nRunning qualitative prompts ({label}) ...")
            for i, prompt in enumerate(PROMPTS):
                print(f"  Prompt {i+1}/10 ...", end=" ", flush=True)
                t0       = time.time()
                response = generate(model, tokenizer, prompt)
                elapsed  = time.time() - t0
                print(f"done ({elapsed:.1f}s)")

                if label == "base":
                    results["qualitative"].append({
                        "prompt":    prompt.split("### Instruction:\n")[1].split("\n\n")[0],
                        "base":      response,
                        "finetuned": "",
                    })
                else:
                    results["qualitative"][i]["finetuned"] = response

            # Free memory before loading next model
            del model
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

        # ── Save results ──────────────────────────────────────────────────
        results_json = RESULTS_DIR / "eval_results.json"
        with open(results_json, "w") as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(str(results_json), artifact_path="evaluation")

        # ── Save qualitative markdown ─────────────────────────────────────
        md_path = RESULTS_DIR / "qualitative_results.md"
        with open(md_path, "w") as f:
            f.write("# Qualitative Evaluation: Base vs Fine-Tuned\n\n")
            f.write(f"| Metric | Base | Fine-Tuned |\n")
            f.write(f"|--------|------|------------|\n")
            f.write(f"| Perplexity (lower=better) | "
                    f"{results['perplexity']['base']:.2f} | "
                    f"{results['perplexity']['fine-tuned']:.2f} |\n\n")
            f.write("---\n\n")
            for i, item in enumerate(results["qualitative"]):
                f.write(f"## Prompt {i+1}\n\n")
                f.write(f"**{item['prompt']}**\n\n")
                f.write(f"### Base Model\n```python\n{item['base']}\n```\n\n")
                f.write(f"### Fine-Tuned Model\n```python\n{item['finetuned']}\n```\n\n")
                f.write("---\n\n")
        mlflow.log_artifact(str(md_path), artifact_path="evaluation")

        # ── Print summary ─────────────────────────────────────────────────
        print(f"""
{'═'*60}
  EVALUATION SUMMARY
{'═'*60}
  Perplexity (lower is better):
    Base model   : {results['perplexity']['base']:.2f}
    Fine-tuned   : {results['perplexity']['fine-tuned']:.2f}
    Improvement  : {((results['perplexity']['base'] - results['perplexity']['fine-tuned']) / results['perplexity']['base'] * 100):.1f}%

  Results saved to: {RESULTS_DIR}
  Logged to MLflow: evaluation run
{'═'*60}
""")


if __name__ == "__main__":
    main()