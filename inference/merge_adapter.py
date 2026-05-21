"""
inference/merge_adapter.py
--------------------------
Loads the base model + LoRA adapter, merges them into a single standalone
HuggingFace model, and saves it to outputs/merged_model/.

Why merge?
- The adapter alone is useless without the base model at inference time
- Merging produces a clean, self-contained model with no PEFT dependency
- The merged model loads faster and is easier to version and serve
"""

import yaml
import torch
import mlflow
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_ID        = cfg["model"]["base_model_id"]
ADAPTER_DIR     = ROOT / cfg["paths"]["adapter_dir"]
MERGED_DIR      = ROOT / cfg["paths"]["merged_model_dir"]
MLFLOW_CFG      = cfg["mlflow"]

MERGED_DIR.mkdir(parents=True, exist_ok=True)


def merge():
    print(f"Loading base model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype    = torch.float16,
        low_cpu_mem_usage = True,
    )

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))

    print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

    print("Merging adapter weights into base model ...")
    model = model.merge_and_unload()
    print("  ✅ Merge complete — PEFT dependency dropped")

    print(f"Saving merged model to: {MERGED_DIR}")
    model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))
    print("  ✅ Merged model saved")

    # ── Log merged model to MLflow ────────────────────────────────────────────
    print("Logging merged model artifact to MLflow ...")
    mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CFG["experiment_name"])

    with mlflow.start_run(run_name="merge-adapter"):
        mlflow.log_param("base_model",   MODEL_ID)
        mlflow.log_param("adapter_dir",  str(ADAPTER_DIR))
        mlflow.log_param("merged_dir",   str(MERGED_DIR))
        mlflow.log_artifacts(str(MERGED_DIR), artifact_path="merged_model")
        print("  ✅ Merged model logged to MLflow")

    print(f"\nDone. Merged model is at: {MERGED_DIR}")
    print("Files:")
    for f in sorted(MERGED_DIR.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:<40} {size_mb:>8.1f} MB")


if __name__ == "__main__":
    merge()