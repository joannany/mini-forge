"""Generate real base/tuned responses for fixture-mode evaluation.

This script is intended for Colab/RunPod/Kaggle after training a LoRA adapter. It
avoids the fragile "run a localhost vLLM server inside the same notebook" path:
responses are generated once, written into `data/eval_set.jsonl`, and the normal
harness runs in fixture mode with `fixture_smoke_test: false`.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from eval.provider import build_messages


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, items: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_unsloth_model(model_id: str, max_seq_length: int):
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Missing Unsloth. In Colab, install with: pip install unsloth trl datasets"
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_answers(model_id: str, items: List[Dict], max_seq_length: int, max_new_tokens: int) -> List[str]:
    model, tokenizer = load_unsloth_model(model_id, max_seq_length)
    outputs = []
    for item in items:
        input_ids = tokenizer.apply_chat_template(
            build_messages(item),
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        answer = tokenizer.decode(
            generated[0][input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()
        outputs.append(answer)

    del model
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval_set.jsonl")
    ap.add_argument("--base-model", default="unsloth/Ministral-8B-Instruct-2410")
    ap.add_argument("--tuned-model", default="training/outputs/tuned-lora")
    ap.add_argument("--out", default=None, help="defaults to overwriting --eval")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--skip-base", action="store_true", help="keep existing base_response values")
    args = ap.parse_args()

    items = load_jsonl(args.eval)
    if not items:
        raise SystemExit(f"no eval items found at {args.eval}")

    if args.skip_base:
        base = [item.get("base_response", "") for item in items]
        if not all(base):
            raise SystemExit("--skip-base requires existing non-empty base_response fields")
    else:
        print(f"generating base responses with {args.base_model}")
        base = generate_answers(args.base_model, items, args.max_seq_length, args.max_new_tokens)

    print(f"generating tuned responses with {args.tuned_model}")
    tuned = generate_answers(args.tuned_model, items, args.max_seq_length, args.max_new_tokens)

    for item, base_response, tuned_response in zip(items, base, tuned):
        item["base_response"] = base_response
        item["tuned_response"] = tuned_response
        item["base_response_source"] = f"model:{args.base_model}"
        item["tuned_response_source"] = f"model:{args.tuned_model}"

    out = args.out or args.eval
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, items)
    print(f"wrote model-generated responses -> {out}")


if __name__ == "__main__":
    main()
