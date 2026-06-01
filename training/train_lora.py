"""QLoRA/SFT training entry point for Mini-Forge.

This is real training code, but it intentionally depends on GPU-oriented packages
that are not installed by default in the lightweight repo. Run it in Colab, Kaggle,
RunPod, Lambda, or another CUDA environment after installing Unsloth + TRL.

Example:
    python -m training.train_lora \
      --train data/synthetic_train.jsonl \
      --base-model unsloth/mistral-7b-instruct-v0.3-bnb-4bit \
      --out training/outputs/tuned-lora \
      --max-steps 60
"""
import argparse
import inspect
import json
from typing import Dict, List


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


def _supports_parameter(callable_obj, name: str) -> bool:
    signature = inspect.signature(callable_obj)
    return (
        name in signature.parameters
        or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    )


def _filtered_kwargs(callable_obj, kwargs: Dict) -> Dict:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/synthetic_train.jsonl")
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
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. In a CUDA environment, install Unsloth/TRL, e.g.\n"
            "  pip install unsloth trl datasets\n"
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

    def format_example(item):
        text = tokenizer.apply_chat_template(
            make_messages(item),
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list([format_example(item) for item in rows])

    sft_config_kwargs = {
        "fp16": args.precision == "fp16",
        "bf16": args.precision == "bf16",
        "output_dir": args.out,
        "dataset_text_field": "text",
        "dataset_num_proc": 1,
        "max_length": args.max_seq_length,
        "max_seq_length": args.max_seq_length,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": 5,
        "save_strategy": "steps",
        "save_steps": max(args.max_steps, 1),
        "optim": "adamw_8bit",
        "warmup_steps": 5,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "report_to": [],
    }
    sft_config = SFTConfig(**_filtered_kwargs(SFTConfig, sft_config_kwargs))

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": sft_config,
    }
    if _supports_parameter(SFTTrainer, "processing_class"):
        trainer_kwargs["processing_class"] = tokenizer
    elif _supports_parameter(SFTTrainer, "tokenizer"):
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise SystemExit("Installed TRL SFTTrainer accepts neither processing_class nor tokenizer.")

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
