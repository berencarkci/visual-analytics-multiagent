"""SFT + QLoRA training for the visual analytics assistant (Task B3/T4).

Trains a LoRA adapter on data/sft_train.jsonl so the model produces valid
chart recommendations from a short prompt, instead of needing the 1260-token
few-shot baseline prompt.

Two deliberate choices worth knowing before reading the code:

  * Completion-only loss. The schema text in the user turn is masked out; the
    loss is computed only on the assistant JSON. Otherwise the model would
    spend capacity learning to *produce* table schemas, which nothing asks of
    it at inference time.

  * 4-bit base, fp16 adapter. Free Colab GPUs (T4) have no bf16 support, so the
    base model is quantized to NF4 and only the adapter trains in fp16. The
    adapter is merged back into an fp16 base for inference, so quantization
    does not follow the model into evaluation.

Usage (from the repo root):
    python training/train_sft.py                       # local / Colab, no push
    python training/train_sft.py --push                # also push to the Hub
    python training/train_sft.py --config other.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

# Qwen2.5 uses the ChatML template; the assistant turn starts with this marker
# and everything before it is masked from the loss:
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
#################################


# Reproducibility:
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
#################################


# Data preparation:
def load_examples(path: str | Path) -> list[dict]:
    """Read the chat-JSONL training file"""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise SystemExit(f"no training examples found in {path}")
    return records


def split_by_source(records: list[dict], val_ratio: float, seed: int) -> tuple[list, list]:
    """Hold out a validation slice, balanced across the three data sources

    Balancing matters because the sources are wildly uneven (mostly template);
    a random split could leave the validation set with no handwritten or
    failure-targeted examples at all, making the eval loss uninformative.
    """
    rng = random.Random(seed)
    buckets: dict[str, list] = {}
    for r in records:
        buckets.setdefault(r.get("meta", {}).get("source", "unknown"), []).append(r)

    train, val = [], []
    for source, items in sorted(buckets.items()):
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio)) if len(items) >= 10 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    return train, val


def to_text_dataset(records: list[dict], tokenizer) -> Dataset:
    """Apply the model's own chat template; training and inference must match"""
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False)
             for r in records]
    return Dataset.from_dict({"text": texts})
#################################


# Model loading:
def load_quantized_model(cfg: dict):
    """4-bit base model, prepared for k-bit LoRA training"""
    compute_dtype = getattr(torch, cfg["bnb_4bit_compute_dtype"])
    quant_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False                 # incompatible with checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["gradient_checkpointing"]
    )
    return model
#################################


# Run report (checklist item: time / compute note):
def write_run_report(cfg: dict, out_dir: Path, elapsed_s: float,
                     n_train: int, n_val: int, history: list[dict]) -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    losses = [h for h in history if "loss" in h]
    evals = [h for h in history if "eval_loss" in h]
    report = {
        "base_model": cfg["base_model"],
        "gpu": gpu,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "examples_train": n_train,
        "examples_val": n_val,
        "epochs": cfg["num_train_epochs"],
        "effective_batch": cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"],
        "learning_rate": cfg["learning_rate"],
        "lora_r": cfg["lora_r"],
        "seed": cfg["seed"],
        "duration_minutes": round(elapsed_s / 60, 1),
        "first_loss": losses[0]["loss"] if losses else None,
        "last_loss": losses[-1]["loss"] if losses else None,
        "eval_loss_per_epoch": [round(e["eval_loss"], 4) for e in evals],
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
#################################


# Entry point:
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/config_sft.yaml")
    ap.add_argument("--push", action="store_true", help="push the adapter to the Hub")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_all_seeds(cfg["seed"])

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found. 4-bit training needs a GPU "
              "(Colab: Runtime > Change runtime type > T4 GPU).")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"               # correct side for training

    records = load_examples(cfg["train_file"])
    train_recs, val_recs = split_by_source(records, cfg["val_ratio"], cfg["seed"])
    train_ds = to_text_dataset(train_recs, tokenizer)
    val_ds = to_text_dataset(val_recs, tokenizer) if val_recs else None
    print(f"examples: {len(train_recs)} train / {len(val_recs)} validation")

    model = load_quantized_model(cfg)
    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    out_dir = Path(cfg["output_dir"])
    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optim"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
        eval_strategy=cfg["eval_strategy"] if val_ds else "no",
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        seed=cfg["seed"],
        fp16=True,                                  # T4: no bf16
        max_seq_length=cfg["max_seq_length"],
        dataset_text_field="text",
        packing=False,                              # required for completion-only loss
        report_to="none",
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE, tokenizer=tokenizer
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        data_collator=collator,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))               # adapter weights + config
    tokenizer.save_pretrained(str(out_dir))
    report = write_run_report(cfg, out_dir, elapsed, len(train_recs),
                              len(val_recs), trainer.state.log_history)

    print(f"\nadapter saved to {out_dir}")
    print(f"duration: {report['duration_minutes']} min on {report['gpu']}")
    print(f"loss: {report['first_loss']} -> {report['last_loss']}"
          f" | eval per epoch: {report['eval_loss_per_epoch']}")

    if args.push or cfg.get("push_to_hub"):
        repo_id = cfg["hub_model_id"]
        trainer.model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print(f"pushed to https://huggingface.co/{repo_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#################################