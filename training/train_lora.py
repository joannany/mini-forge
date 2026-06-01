"""QLoRA training entry point for Mini-Forge.

This script deliberately avoids TRL's SFTTrainer. Current Colab/Unsloth/TRL stacks
can fail while TRL tokenizes datasets with multiprocessing, even for tiny datasets.
Mini-Forge pre-renders chat text, pre-tokenizes it once, and trains the PEFT model with
`transformers.Trainer` directly.

Example:
    python -m training.train_lora \
      --train data/synthetic_train.clean.jsonl \
      --base-model unsloth/mistral-7b-instruct-v0.3-bnb-4bit \
      --out training/outputs/tuned-lora \
      --max-steps 80
"""
import argparse
import json
import os
from typing import Dict, List

os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


SYSTEM_PROMPT = """You are a compliance assistant for regulated enterprise policy QA.
Answer only from the provided policy excerpt. If the excerpt does not answer the
question, say that it is not covered by the provided policy and do not invent a rule.
Use this structure:
Answer: ...
Source: ...
Risk note: ...
"""


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_messages(item: Dict) -> List[Dict[str, str]]:
    policy = item.get("reference") or "No applicable policy excerpt was provided."
    user = (
        f"POLICY EXCERPT:\n{policy}\n\n"
        f"QUESTION:\n{item['question']}\n\n"
        "Return the compliant answer."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": item["response"]},
    ]


def render_training_text(tokenizer, item: Dict) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            make_messages(item),
            tokenize=False,
            add_generation_prompt=False,
        )
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in make_messages(item))


def tokenize_rows(tokenizer, rows: List[Dict], max_seq_length: int) -> List[Dict]:
    tokenized = []
    for item in rows:
        encoded = tokenizer(
            render_training_text(tokenizer, item),
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": list(input_ids),
            }
        )
    return tokenized


def make_collator(tokenizer):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    def collate(batch: List[Dict]):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("training requires torch") from exc

        max_len = max(len(row["input_ids"]) for row in batch)
        input_ids, attention_mask, labels = [], [], []
        for row in batch:
            pad = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_token_id] * pad)
            attention_mask.append(row["attention_mask"] + [0] * pad)
            labels.append(row["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/synthetic_train.clean.jsonl")
    ap.add_argument("--base-model", default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    ap.add_argument("--out", default="training/outputs/tuned-lora")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument(
        "--precision",
        choices=["fp16", "bf16"],
        default="fp16",
        help="T4-safe default is fp16. Use bf16 only on Ampere+ GPUs.",
    )
    args = ap.parse_args()

    try:
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. In a CUDA environment, install:\n"
            "  pip install unsloth datasets transformers accelerate\n"
            "Then rerun this script."
        ) from exc

    rows = load_jsonl(args.train)
    if not rows:
        raise SystemExit(f"no training rows found at {args.train}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    tokenized = tokenize_rows(tokenizer, rows, args.max_seq_length)
    train_dataset = Dataset.from_list(tokenized)

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="steps",
        save_steps=max(args.max_steps, 1),
        optim="adamw_8bit",
        warmup_steps=5,
        lr_scheduler_type="linear",
        seed=3407,
        report_to=[],
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_collator(tokenizer),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
