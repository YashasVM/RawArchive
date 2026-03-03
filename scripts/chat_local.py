from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "data" / "models"
CACHE_DIR = PROJECT_ROOT / "data" / ".adapter_cache"


def resolve_model_meta(model_id: str) -> dict:
    model_path = MODELS_DIR / f"{model_id}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {model_path}")
    return json.loads(model_path.read_text(encoding="utf-8"))


def parse_adapter_uri(adapter_uri: str) -> Path:
    if adapter_uri.startswith("local://"):
        return Path(adapter_uri[len("local://") :])
    return Path(adapter_uri)


def resolve_adapter_path(adapter_path: Path, model_id: str) -> Path:
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    if adapter_path.is_dir():
        return adapter_path

    if adapter_path.suffix.lower() != ".zip":
        raise ValueError(f"Adapter must be a directory or .zip file: {adapter_path}")

    target_dir = CACHE_DIR / model_id
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(adapter_path, "r") as archive:
        archive.extractall(target_dir)
    return target_dir / "outputs" / "adapter" if (target_dir / "outputs" / "adapter").exists() else target_dir


def load_model_and_tokenizer(base_model: str, adapter_dir: Path, use_4bit: bool):
    def _build_quant_config():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    quant_config = _build_quant_config() if use_4bit else None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
    except Exception:
        if not use_4bit:
            raise
        print("4-bit load failed; retrying without 4-bit quantization.")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_prompt_messages(system_prompt: str, chat_history: list[dict], user_text: str) -> list[dict]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_text})
    return messages


def generate_reply(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local chat with a registered LoRA adapter")
    parser.add_argument("--model-id", default="mdl_bb8e7abb4c", help="Registered model id from data/models")
    parser.add_argument("--base-model", default="", help="Override base model from metadata")
    parser.add_argument("--adapter-uri", default="", help="Override adapter uri from metadata")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
    parser.add_argument(
        "--system-prompt",
        default="Reply in the user's texting style: concise, casual, and context-aware.",
    )
    args = parser.parse_args()

    meta = resolve_model_meta(args.model_id)
    adapter_uri = args.adapter_uri or str(meta["adapter_uri"])
    base_model = args.base_model or str(meta["base_model"])

    adapter_path = parse_adapter_uri(adapter_uri)
    adapter_dir = resolve_adapter_path(adapter_path, args.model_id)

    print(f"Loading base model: {base_model}")
    print(f"Using adapter: {adapter_dir}")
    model, tokenizer = load_model_and_tokenizer(base_model, adapter_dir, use_4bit=not args.no_4bit)

    print("\nInteractive chat ready. Type /exit to quit, /reset to clear history.\n")
    history: list[dict] = []
    while True:
        user_text = input("you> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit"}:
            break
        if user_text.lower() == "/reset":
            history.clear()
            print("history cleared")
            continue

        messages = build_prompt_messages(args.system_prompt, history, user_text)
        reply = generate_reply(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"bot> {reply}\n")
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
