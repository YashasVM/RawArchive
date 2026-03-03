from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def download_bundle(url: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / "bundle.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)
    return destination


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_list(records)


def build_model(base_model: str, use_4bit: bool):
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    return model


def tokenize_function(example: dict, tokenizer, max_length: int) -> dict:
    prompt = example["input"].strip()
    answer = example["output"].strip()

    prompt_text = f"{prompt}\n"
    full_text = f"{prompt_text}{answer}"

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    labels = input_ids.copy()
    prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
    for idx in range(prompt_len):
        labels[idx] = -100
    for idx, mask in enumerate(attention_mask):
        if mask == 0:
            labels[idx] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA training entrypoint for personalized chat style")
    parser.add_argument("--bundle-url", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        bundle_dir = download_bundle(args.bundle_url, workspace)

        train_records = load_jsonl(bundle_dir / "train.jsonl")
        val_records = load_jsonl(bundle_dir / "val.jsonl")
        train_config = yaml.safe_load((bundle_dir / "train_config.yaml").read_text(encoding="utf-8"))

        if args.max_train_samples and args.max_train_samples > 0:
            train_records = train_records[: args.max_train_samples]
        if args.max_val_samples and args.max_val_samples > 0:
            val_records = val_records[: args.max_val_samples]

        if args.epochs is not None:
            train_config["epochs"] = args.epochs
        if args.seq_len is not None:
            train_config["seq_len"] = args.seq_len
        if args.batch_size is not None:
            train_config["batch_size"] = args.batch_size
        if args.grad_accum is not None:
            train_config["gradient_accumulation_steps"] = args.grad_accum
        if args.learning_rate is not None:
            train_config["learning_rate"] = args.learning_rate

        print(
            f"Train samples={len(train_records)} val samples={len(val_records)} "
            f"seq_len={train_config.get('seq_len')} batch={train_config.get('batch_size')} "
            f"grad_accum={train_config.get('gradient_accumulation_steps')} epochs={train_config.get('epochs')}"
        )

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_dataset = build_dataset(train_records)
        val_dataset = build_dataset(val_records)

        max_length = int(train_config.get("seq_len", 1024))
        train_dataset = train_dataset.map(
            lambda ex: tokenize_function(ex, tokenizer, max_length),
            remove_columns=train_dataset.column_names,
        )
        val_dataset = val_dataset.map(
            lambda ex: tokenize_function(ex, tokenizer, max_length),
            remove_columns=val_dataset.column_names,
        )

        use_4bit = bool(train_config.get("use_4bit", True))
        model = build_model(args.base_model, use_4bit=use_4bit)

        lora_cfg = train_config.get("lora", {})
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)

        args_train = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            per_device_train_batch_size=int(train_config.get("batch_size", 4)),
            gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 8)),
            num_train_epochs=float(train_config.get("epochs", 2)),
            learning_rate=float(train_config.get("learning_rate", 2e-4)),
            logging_steps=20,
            eval_strategy="steps" if len(val_dataset) > 0 else "no",
            eval_steps=100 if len(val_dataset) > 0 else None,
            save_steps=100,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to="none",
            lr_scheduler_type=str(train_config.get("scheduler", "cosine")),
        )

        trainer = Trainer(
            model=model,
            args=args_train,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        )

        trainer.train()
        eval_metrics = trainer.evaluate() if len(val_dataset) > 0 else {}

        adapter_dir = output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        (output_dir / "metrics.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
        print("Training complete")
        print(f"Adapter exported to: {adapter_dir}")
        print(f"Metrics: {json.dumps(eval_metrics)}")


if __name__ == "__main__":
    main()
