"""DPO training on top of the SFT adapter.

Direct Preference Optimisation nudges the model toward the chosen answer and
away from the rejected one in each pair, while a KL term keeps it from drifting
far from where SFT left it.

Two choices worth knowing before reading the code:

  * No separate reference model. The SFT adapter is loaded as trainable and TRL
    disables it to obtain reference log-probabilities. A second copy of a 3B
    model would not leave room on a free T4, and the reference is exactly "the
    model before this run" anyway.

  * A much smaller learning rate than SFT (5e-6 against 2e-4). Preference data
    is a correction, not a curriculum: 430 pairs at SFT's rate would overwrite
    the behaviour the 1884 SFT examples established.

The resulting adapter therefore carries SFT and DPO together. The two effects
are still separable in evaluation, by running the SFT adapter and this one side
by side.

Usage (from the repo root):
    python training/train_dpo.py
    python training/train_dpo.py --config training/config_dpo.yaml
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
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer
#################################


# Reproducibility:
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
#################################


# Data:
def load_pairs(path: str | Path) -> list[dict]:
    """Read the pair file written by evaluation/make_dpo_pairs.py"""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no preference pairs found in {path}")
    return rows


def to_dpo_dataset(rows: list[dict]) -> Dataset:
    """TRL's conversational DPO format: every field is a message list

    The pair file stores completions as plain JSON strings because that is what
    the agent emits; here they become single assistant turns so the trainer can
    apply the chat template the same way it does for the prompt.
    """
    return Dataset.from_list([
        {"prompt": r["prompt"],
         "chosen": [{"role": "assistant", "content": r["chosen"]}],
         "rejected": [{"role": "assistant", "content": r["rejected"]}]}
        for r in rows
    ])


def split_by_format(rows: list[dict], val_ratio: float, seed: int) -> tuple[list, list]:
    """Hold out a validation slice with every output format represented

    The formats are uneven (data_analyst 183, supervisor 157, visualization 90)
    and they behave differently: a random split can leave one of them out and
    make the eval loss blind to exactly the skill that regressed.
    """
    rng = random.Random(seed)
    buckets: dict[str, list] = {}
    for r in rows:
        buckets.setdefault(r.get("meta", {}).get("format", "unknown"), []).append(r)

    train, val = [], []
    for fmt, items in sorted(buckets.items()):
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio)) if len(items) >= 10 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    return train, val
#################################


# Model:
def load_model(cfg: dict):
    """4-bit base + the SFT adapter, loaded trainable so DPO continues from it"""
    compute_dtype = getattr(torch, cfg["bnb_4bit_compute_dtype"])
    quant_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=quant_config,
        device_map="auto", torch_dtype=compute_dtype,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=cfg["gradient_checkpointing"])

    model = PeftModel.from_pretrained(base, cfg["sft_adapter"], is_trainable=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable / 1e6:.1f}M")
    return model
#################################


# Run report:
def write_run_report(cfg: dict, out_dir: Path, elapsed_s: float,
                     n_train: int, n_val: int, history: list[dict]) -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    losses = [h for h in history if "loss" in h]
    evals = [h for h in history if "eval_loss" in h]
    # DPO logs two diagnostics worth keeping: how often the chosen answer scores
    # higher than the rejected one, and by how much.
    acc = [h["rewards/accuracies"] for h in history if "rewards/accuracies" in h]
    margins = [h["rewards/margins"] for h in history if "rewards/margins" in h]

    report = {
        "base_model": cfg["base_model"],
        "sft_adapter": cfg["sft_adapter"],
        "gpu": gpu,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "pairs_train": n_train,
        "pairs_val": n_val,
        "epochs": cfg["num_train_epochs"],
        "beta": cfg["beta"],
        "learning_rate": cfg["learning_rate"],
        "effective_batch": cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"],
        "seed": cfg["seed"],
        "duration_minutes": round(elapsed_s / 60, 1),
        "first_loss": losses[0]["loss"] if losses else None,
        "last_loss": losses[-1]["loss"] if losses else None,
        "eval_loss_per_epoch": [round(e["eval_loss"], 4) for e in evals],
        "reward_accuracy_first": round(acc[0], 3) if acc else None,
        "reward_accuracy_last": round(acc[-1], 3) if acc else None,
        "reward_margin_last": round(margins[-1], 3) if margins else None,
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
#################################


# Entry point:
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/config_dpo.yaml")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_all_seeds(cfg["seed"])

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found. 4-bit training needs a GPU.")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_pairs(cfg["train_file"])
    train_rows, val_rows = split_by_format(rows, cfg["val_ratio"], cfg["seed"])
    train_ds = to_dpo_dataset(train_rows)
    val_ds = to_dpo_dataset(val_rows) if val_rows else None
    print(f"pairs: {len(train_rows)} train / {len(val_rows)} validation")

    model = load_model(cfg)

    out_dir = Path(cfg["output_dir"])
    dpo_config = DPOConfig(
        output_dir=str(out_dir),
        beta=cfg["beta"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optim"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
        eval_strategy=cfg["eval_strategy"] if val_ds else "no",
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=cfg.get("load_best_model_at_end", False) and val_ds is not None,
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg.get("greater_is_better", False),
        seed=cfg["seed"],
        fp16=True,
        max_length=cfg["max_length"],
        max_prompt_length=cfg["max_prompt_length"],
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                 # the SFT adapter is disabled to act as reference
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    report = write_run_report(cfg, out_dir, elapsed, len(train_rows),
                              len(val_rows), trainer.state.log_history)

    print(f"\nadapter saved to {out_dir}")
    print(f"duration: {report['duration_minutes']} min on {report['gpu']}")
    print(f"loss: {report['first_loss']} -> {report['last_loss']}"
          f" | eval per epoch: {report['eval_loss_per_epoch']}")
    print(f"reward accuracy: {report['reward_accuracy_first']} -> "
          f"{report['reward_accuracy_last']} | margin: {report['reward_margin_last']}")

    if args.push or cfg.get("push_to_hub"):
        repo_id = cfg["hub_model_id"]
        trainer.model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print(f"pushed to https://huggingface.co/{repo_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#################################