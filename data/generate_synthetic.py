"""Generate synthetic compliance instruction data from local policy documents.

This script runs without a generator model. It extracts policy-like sentences from
public/plain-text docs and creates behavior-focused examples for SFT.
The deterministic examples can be used as seed data or human-review targets for a
separate generator-based data pipeline.
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List

KINDS = ("answerable", "benign_in_scope", "out_of_scope", "unanswerable")


def read_documents(docs_dir: str) -> List[Dict[str, str]]:
    docs = []
    root = Path(docs_dir)
    if not root.exists():
        return docs
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append({"name": path.stem, "text": text})
    return docs


def chunk_sentences(text: str, max_chars: int = 900) -> Iterable[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    current = []
    size = 0
    for sentence in sentences:
        if not sentence:
            continue
        if size + len(sentence) > max_chars and current:
            yield " ".join(current)
            current, size = [], 0
        current.append(sentence)
        size += len(sentence)
    if current:
        yield " ".join(current)


def _topic(excerpt: str) -> str:
    words = [
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z-]{3,}", excerpt)
        if w.lower() not in {
            "this", "that", "with", "from", "shall", "must", "will", "policy",
            "data", "user", "users", "information", "company", "customer",
        }
    ]
    return " ".join(words[:4]) if words else "this policy requirement"


def _citation(doc_name: str, idx: int) -> str:
    return f"{doc_name} excerpt {idx}"


def make_item(doc_name: str, excerpt: str, idx: int, kind: str) -> Dict:
    topic = _topic(excerpt)
    source = _citation(doc_name, idx)
    if kind == "answerable":
        question = f"What does the policy say about {topic}?"
        response = (
            f"Answer: The provided policy states: {excerpt}\n"
            f"Source: {source}\n"
            "Risk note: Apply this only to the workflow covered by the cited policy."
        )
        return _item(doc_name, idx, kind, question, excerpt, response, source)
    if kind == "benign_in_scope":
        question = f"Summarize the policy requirement for {topic} in plain language."
        response = (
            f"Answer: In plain language, the covered requirement is: {excerpt}\n"
            f"Source: {source}\n"
            "Risk note: This is a summary, not a replacement for the policy text."
        )
        return _item(doc_name, idx, kind, question, excerpt, response, source)
    if kind == "out_of_scope":
        question = f"Does this policy authorize an unrelated exception for {topic} in overseas tax filings?"
        response = (
            "Answer: This is not covered by the provided policy excerpt, so I cannot "
            "determine an authorization from this source.\n"
            f"Source: {source}\n"
            "Risk note: Route the question to the appropriate policy owner."
        )
        return _item(doc_name, idx, kind, question, "", response, source)

    question = f"What will the policy requirement for {topic} be five years from now?"
    response = (
        "Answer: The future policy requirement is not specified in the provided excerpt, "
        "so I cannot predict or invent it.\n"
        f"Source: {source}\n"
        "Risk note: Use the latest approved policy before making a deployment decision."
    )
    return _item(doc_name, idx, kind, question, "", response, source)


def _item(doc_name: str, idx: int, kind: str, question: str, reference: str, response: str, source: str) -> Dict:
    return {
        "id": f"{doc_name}-{idx}-{kind}",
        "question": question,
        "reference": reference,
        "kind": kind,
        "response": response,
        "required_sections": ["Answer:", "Source:", "Risk note:"],
        "must_cite": True,
        "source": source,
    }


def fallback_docs() -> List[Dict[str, str]]:
    return [
        {
            "name": "sample_retention_policy",
            "text": (
                "Transaction records are retained for 7 years per financial regulation. "
                "Access to production customer data must be logged and reviewed quarterly. "
                "Employees may not export customer personal data to unmanaged devices."
            ),
        },
        {
            "name": "sample_acceptable_use",
            "text": (
                "Acceptable use covers prohibited content, security, and access rules. "
                "Users must not share credentials or bypass access controls. "
                "Security incidents must be escalated to the response team within 24 hours."
            ),
        },
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/policy_docs")
    ap.add_argument("--out", default="data/synthetic_train.jsonl")
    ap.add_argument("--eval-out", default="data/eval_set.jsonl")
    ap.add_argument("--per-doc", type=int, default=12)
    ap.add_argument("--eval-ratio", type=float, default=0.25)
    args = ap.parse_args()

    docs = read_documents(args.docs) or fallback_docs()
    items = []
    for doc in docs:
        chunks = list(chunk_sentences(doc["text"]))[: max(1, args.per_doc)]
        for idx, excerpt in enumerate(chunks, start=1):
            for kind in KINDS:
                items.append(make_item(doc["name"], excerpt, idx, kind))

    eval_every = max(2, round(1 / args.eval_ratio)) if args.eval_ratio > 0 else 0
    train, eval_items = [], []
    for idx, item in enumerate(items):
        if eval_every and idx % eval_every == 0:
            eval_item = dict(item)
            # Fixture mode is a plumbing smoke test, not a baseline-vs-tuned model
            # evaluation. The same gold response is used for both sides so the
            # default pipeline cannot report a usefulness gain from missing baseline
            # responses. Real baseline responses are written by the response
            # generation scripts or generated through non-fixture provider modes.
            eval_item["base_response"] = item["response"]
            eval_item["tuned_response"] = item["response"]
            eval_item["base_response_source"] = "gold_fixture"
            eval_item["tuned_response_source"] = "gold_fixture"
            eval_items.append(eval_item)
        else:
            train.append(item)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(args.eval_out, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"wrote {len(train)} train items -> {args.out}")
    print(f"wrote {len(eval_items)} eval items -> {args.eval_out}")


if __name__ == "__main__":
    main()
