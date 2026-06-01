"""Generate real before/after responses without LoRA training.

This script evaluates a behavioral prompt intervention through the same Mini-Forge
gate:
- baseline: bare/minimal prompt
- tuned: compliant Mini-Forge system prompt

It is not a fine-tune. It is a fast, honest way to produce real model-generated gate
numbers when the LoRA path is blocked by GPU/library issues.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from eval.provider import SYSTEM_PROMPT, build_messages


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, items: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_bare_messages(item: Dict) -> List[Dict[str, str]]:
    policy = item.get("reference") or "No applicable policy excerpt was provided."
    return [
        {
            "role": "user",
            "content": (
                f"Policy:\n{policy}\n\n"
                f"Question:\n{item.get('question', '')}\n\n"
                "Answer the question."
            ),
        }
    ]


def build_compliant_messages(item: Dict) -> List[Dict[str, str]]:
    return build_messages(item)


def load_unsloth_model(model_id: str, max_seq_length: int):
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Missing Unsloth. In Colab, install with: pip install unsloth datasets transformers accelerate"
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.decode(
        generated[0][input_ids.shape[-1]:],
        skip_special_tokens=True,
    ).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval_set.jsonl")
    ap.add_argument("--model", default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    ap.add_argument("--out", default=None, help="defaults to overwriting --eval")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    items = load_jsonl(args.eval)
    if not items:
        raise SystemExit(f"no eval items found at {args.eval}")

    print(f"loading model {args.model}")
    model, tokenizer = load_unsloth_model(args.model, args.max_seq_length)

    for item in items:
        item["base_response"] = generate(
            model,
            tokenizer,
            build_bare_messages(item),
            args.max_new_tokens,
        )
        item["tuned_response"] = generate(
            model,
            tokenizer,
            build_compliant_messages(item),
            args.max_new_tokens,
        )
        item["base_response_source"] = f"prompt:bare:{args.model}"
        item["tuned_response_source"] = f"prompt:compliant:{args.model}"

    out = args.out or args.eval
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, items)
    print(f"wrote prompt-intervention responses -> {out}")
    print("Run the gate with generation.mode=fixture and fixture_smoke_test=false.")
    print(f"Compliant system prompt used:\n{SYSTEM_PROMPT}")


if __name__ == "__main__":
    main()
