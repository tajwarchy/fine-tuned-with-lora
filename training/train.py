"""
training/train.py
-----------------
LoRA fine-tuning on M1 CPU with MLflow logging.

Key M1 constraints honoured:
  - No bitsandbytes 4-bit quantization (CUDA-only)
  - Model loaded in float16, cast to float32 for stable CPU training
  - num_workers=0 everywhere
  - Batch size 1 + gradient accumulation 8 = effective batch 8
  - Designed to run overnight (~6-10 hours)
"""

import json
import os
import math
import time
import yaml
import torch
import mlflow
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_ID      = cfg["model"]["base_model_id"]
LORA_CFG      = cfg["lora"]
TRAIN_CFG     = cfg["training"]
DATASET_CFG   = cfg["dataset"]
MLFLOW_CFG    = cfg["mlflow"]
PATHS_CFG     = cfg["paths"]

DEVICE        = torch.device(TRAIN_CFG["device"])   # "cpu" for M1 training
OUTPUT_DIR    = ROOT / PATHS_CFG["adapter_dir"]
TRAIN_JSON    = ROOT / "data" / "train.json"
MAX_LENGTH    = DATASET_CFG["max_length"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────
class InstructionDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        with open(path) as f:
            self.examples = json.load(f)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]["text"]
        enc  = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        # For causal LM: labels = input_ids, pad tokens masked with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── Model + LoRA ──────────────────────────────────────────────────────────────
def load_model_and_tokenizer():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {MODEL_ID}  (float32 on CPU)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,   # float32 for stable CPU training
        low_cpu_mem_usage=True,
    )
    model.to(DEVICE)

    lora_config = LoraConfig(
        r               = LORA_CFG["rank"],
        lora_alpha      = LORA_CFG["alpha"],
        lora_dropout    = LORA_CFG["dropout"],
        target_modules  = LORA_CFG["target_modules"],
        bias            = LORA_CFG["bias"],
        task_type       = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    model, tokenizer = load_model_and_tokenizer()

    dataset    = InstructionDataset(TRAIN_JSON, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size  = TRAIN_CFG["per_device_train_batch_size"],
        shuffle     = True,
        num_workers = 0,   # must be 0 on M1
        pin_memory  = False,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = TRAIN_CFG["learning_rate"],
        weight_decay = TRAIN_CFG["weight_decay"],
    )

    num_epochs        = TRAIN_CFG["num_train_epochs"]
    grad_accum_steps  = TRAIN_CFG["gradient_accumulation_steps"]
    total_steps       = math.ceil(len(dataloader) / grad_accum_steps) * num_epochs
    warmup_steps      = int(total_steps * TRAIN_CFG["warmup_ratio"])

    lr_scheduler = get_scheduler(
        name              = TRAIN_CFG["lr_scheduler_type"],
        optimizer         = optimizer,
        num_warmup_steps  = warmup_steps,
        num_training_steps= total_steps,
    )

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CFG["experiment_name"])

    with mlflow.start_run() as run:
        # Log all hyperparameters
        mlflow.log_params({
            "base_model":              MODEL_ID,
            "lora_rank":               LORA_CFG["rank"],
            "lora_alpha":              LORA_CFG["alpha"],
            "lora_target_modules":     str(LORA_CFG["target_modules"]),
            "learning_rate":           TRAIN_CFG["learning_rate"],
            "batch_size":              TRAIN_CFG["per_device_train_batch_size"],
            "gradient_accumulation":   grad_accum_steps,
            "effective_batch_size":    TRAIN_CFG["per_device_train_batch_size"] * grad_accum_steps,
            "num_epochs":              num_epochs,
            "max_length":              MAX_LENGTH,
            "train_samples":           len(dataset),
        })

        print(f"\nMLflow run ID: {run.info.run_id}")
        print(f"Starting training — {len(dataloader)} steps/epoch, "
              f"{grad_accum_steps} grad accum → {total_steps} optimizer steps total\n")

        global_step    = 0
        optimizer_step = 0
        model.train()
        optimizer.zero_grad()

        for epoch in range(num_epochs):
            epoch_loss  = 0.0
            epoch_start = time.time()
            print(f"\n{'═'*70}")
            print(f"  EPOCH {epoch+1}/{num_epochs}  —  {len(dataloader)} steps")
            print(f"{'═'*70}")

            for step, batch in enumerate(dataloader):
                step_start     = time.time()
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["labels"].to(DEVICE)

                outputs = model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    labels         = labels,
                )
                loss = outputs.loss / grad_accum_steps
                loss.backward()

                raw_loss    = outputs.loss.item()
                epoch_loss += raw_loss
                global_step += 1

                # ── Live progress line (every step) ──────────────────────────
                step_time    = time.time() - step_start
                elapsed      = time.time() - epoch_start
                steps_done   = step + 1
                steps_left   = len(dataloader) - steps_done
                eta_sec      = steps_left * (elapsed / steps_done)
                eta_str      = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
                avg_loss     = epoch_loss / steps_done
                pct          = steps_done / len(dataloader) * 100
                bar_filled   = int(pct / 2)
                bar          = "█" * bar_filled + "░" * (50 - bar_filled)

                print(
                    f"\r[{bar}] {pct:5.1f}%  "
                    f"step {global_step:>6}/{len(dataloader)}  "
                    f"loss {raw_loss:.4f}  avg {avg_loss:.4f}  "
                    f"⏱ {step_time:.1f}s/step  ETA {eta_str}",
                    end="", flush=True
                )

                # Log to MLflow every logging_steps
                if global_step % TRAIN_CFG["logging_steps"] == 0:
                    mlflow.log_metric("train_loss", raw_loss, step=global_step)

                # Optimizer step every grad_accum_steps
                if global_step % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    optimizer_step += 1
                    mlflow.log_metric("learning_rate", lr_scheduler.get_last_lr()[0], step=optimizer_step)

                # Save checkpoint every save_steps (optimizer steps)
                if optimizer_step > 0 and optimizer_step % TRAIN_CFG["save_steps"] == 0:
                    print()  # newline before checkpoint message
                    ckpt_dir = OUTPUT_DIR / f"checkpoint-{optimizer_step}"
                    model.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))
                    print(f"  ✅ Checkpoint saved → {ckpt_dir}")

            avg_epoch_loss = epoch_loss / len(dataloader)
            elapsed_str    = time.strftime("%H:%M:%S", time.gmtime(time.time() - epoch_start))
            print(f"\n{'─'*70}")
            print(f"  Epoch {epoch+1} complete — avg loss: {avg_epoch_loss:.4f}  |  time: {elapsed_str}")
            print(f"{'─'*70}\n")
            mlflow.log_metric("epoch_avg_loss", avg_epoch_loss, step=epoch + 1)

        # ── Final save ────────────────────────────────────────────────────────
        print(f"Saving final adapter → {OUTPUT_DIR}")
        model.save_pretrained(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))

        # Log adapter as MLflow artifact
        mlflow.log_artifacts(str(OUTPUT_DIR), artifact_path="adapter")
        print(f"\n✅ Training complete. Adapter saved and logged to MLflow.")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   Adapter path: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()