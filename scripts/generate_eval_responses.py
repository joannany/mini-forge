"""Generate real base/tuned responses for fixture-mode evaluation.

This script supports notebook and hosted-GPU environments after training a LoRA
adapter. It precomputes base and tuned responses, writes them into
`data/eval_set.jsonl`, and lets the harness run in fixture mode with
`fixture_smoke_test: false`.
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


def _adapter_base_model(adapter_dir: Path, fallback: str) -> str:
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        return fallback
    try:
        data = json.loads(adapter_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback
    return data.get("base_model_name_or_path") or fallback


def load_unsloth_model(model_id: str, max_seq_length: int, adapter_base_model: str = None):
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Missing Unsloth. In Colab, install with: pip install unsloth datasets transformers accelerate"
        ) from exc

    model_path = Path(model_id)
    if model_path.exists() and (model_path / "adapter_config.json").exists():
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("Loading a LoRA adapter directory requires peft.") from exc

        base_model = _adapter_base_model(model_path, adapter_base_model)
        if not base_model:
            raise SystemExit(
                f"{model_id} looks like a LoRA adapter, but no base model was provided. "
                "Pass --adapter-base-model."
            )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(model, model_id)
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_answers(model_id: str,
                     items: List[Dict],
                     max_seq_length: int,
                     max_new_tokens: int,
                     adapter_base_model: str = None) -> List[str]:
    model, tokenizer = load_unsloth_model(model_id, max_seq_length, adapter_base_model)
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
    ap.add_argument("--base-model", default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    ap.add_argument("--tuned-model", default="training/outputs/tuned-lora")
    ap.add_argument(
        "--adapter-base-model",
        default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        help="Base model to load before applying a local LoRA adapter directory.",
    )
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
    tuned = generate_answers(
        args.tuned_model,
        items,
        args.max_seq_length,
        args.max_new_tokens,
        adapter_base_model=args.adapter_base_model,
    )

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
