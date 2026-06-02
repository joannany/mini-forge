"""
data/prepare_data.py

Validation, dedupe, PII scan, and train/eval leakage check for the synthetic
instruction data. The leakage check verifies that eval questions do not also appear
in the training set.

Usage:
    python -m data.prepare_data --train data/synthetic_train.jsonl --eval data/eval_set.jsonl
"""
import argparse
import hashlib
import json
import os
import re

# Minimal PII patterns — extend or swap for Presidio for production-grade scanning.
PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "krrn_like": re.compile(r"\b\d{6}-\d{7}\b"),   # KR resident-reg-number shape
    "phone": re.compile(r"\b\d{2,4}-\d{3,4}-\d{4}\b"),
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def dedupe(items):
    seen, out = set(), []
    for it in items:
        key = hashlib.md5(normalize(it["question"]).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out, len(items) - len(out)


def scan_pii(items):
    hits = []
    for it in items:
        blob = " ".join(str(v) for v in it.values())
        for name, pat in PII_PATTERNS.items():
            if pat.search(blob):
                hits.append((it.get("id", "?"), name))
    return hits


def validate_schema(items, name):
    required = {"id", "question", "kind"}
    valid_kinds = {"answerable", "benign_in_scope", "out_of_scope", "unanswerable", "tool"}
    errors = []
    for idx, item in enumerate(items, start=1):
        missing = required - set(item)
        if missing:
            errors.append(f"{name}:{idx} missing {sorted(missing)}")
        if item.get("kind") not in valid_kinds:
            errors.append(f"{name}:{idx} invalid kind {item.get('kind')!r}")
        if item.get("kind") in {"answerable", "benign_in_scope"} and not item.get("reference"):
            errors.append(f"{name}:{idx} answerable item has empty reference")
    return errors


def leakage(train, eval_):
    train_keys = {hashlib.md5(normalize(t["question"]).encode()).hexdigest() for t in train}
    leaked = [e.get("id", "?") for e in eval_
              if hashlib.md5(normalize(e["question"]).encode()).hexdigest() in train_keys]
    return leaked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", default=None, help="optional path for cleaned/deduped train JSONL")
    ap.add_argument("--allow-pii", action="store_true")
    args = ap.parse_args()

    train, eval_ = load_jsonl(args.train), load_jsonl(args.eval)
    errors = validate_schema(train, "train") + validate_schema(eval_, "eval")
    if errors:
        print("SCHEMA errors:")
        for err in errors[:20]:
            print("  -", err)
        raise SystemExit("ABORT: schema validation failed")

    train, dropped = dedupe(train)
    pii = scan_pii(train) + scan_pii(eval_)
    leaked = leakage(train, eval_)

    print(f"train items: {len(train)} (deduped {dropped})")
    print(f"eval items:  {len(eval_)}")
    print(f"PII hits:    {len(pii)} -> {pii[:5]}{' ...' if len(pii) > 5 else ''}")
    print(f"LEAKAGE:     {len(leaked)} eval items found in train -> {leaked[:5]}")
    if pii and not args.allow_pii:
        raise SystemExit("ABORT: PII-like pattern detected. Review or pass --allow-pii.")
    if leaked:
        raise SystemExit("ABORT: eval/train leakage detected. Fix before training.")
    if args.out:
        write_jsonl(args.out, train)
        print(f"wrote cleaned train set -> {args.out}")


if __name__ == "__main__":
    main()
