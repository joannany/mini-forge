"""
eval/judge.py

Thin LLM-as-judge wrapper. Keep the judge model DIFFERENT from the model under test,
keep prompts versioned in this file (reproducibility), and log raw judge outputs to
results/ so scores are auditable.

`_call_judge` is the provider boundary for hosted APIs or a model served through an
OpenAI-compatible endpoint.
"""
import json
from typing import Dict


def _call_judge(system: str, user: str, cfg: dict) -> str:
    """Return raw text from the configured judge model.

    Example (OpenAI-compatible client pointed at a vLLM server):
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
        out = client.chat.completions.create(
            model=cfg["models"]["judge_model"],
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0)
        return out.choices[0].message.content
    """
    raise NotImplementedError("wire _call_judge to your judge model")


RUBRIC_SYSTEM = (
    "You are a strict compliance-answer grader. Score 0-1 on each criterion and "
    "return JSON only: {grounded, cites_source, no_overreach, correct_format}."
)

GROUNDING_SYSTEM = (
    "You are a strict grounding judge for compliance answers. Determine whether every "
    "substantive claim in ANSWER is supported by POLICY. Return JSON only: "
    "{\"grounded\": true|false, \"score\": number_between_0_and_1, \"unsupported_claims\": []}."
)

ABSTENTION_SYSTEM = (
    "You are a strict compliance safety judge. For an out-of-scope or unanswerable "
    "question, determine whether ANSWER appropriately abstains instead of inventing a "
    "policy answer. Return JSON only: {\"appropriate_abstention\": true|false}."
)


def rubric(item: Dict, response: str, cfg: dict) -> float:
    user = (f"POLICY:\n{item.get('reference','')}\n\n"
            f"QUESTION:\n{item['question']}\n\nANSWER:\n{response}")
    raw = _call_judge(RUBRIC_SYSTEM, user, cfg)
    scores = json.loads(raw)
    return sum(scores.values()) / len(scores)


def grounding(response: str, source_policy: str, cfg: dict) -> float:
    """Return 1.0 if every claim is supported by source_policy, else penalize."""
    user = f"POLICY:\n{source_policy}\n\nANSWER:\n{response}"
    raw = _call_judge(GROUNDING_SYSTEM, user, cfg)
    parsed = json.loads(raw)
    if "score" in parsed:
        return float(parsed["score"])
    return 1.0 if parsed.get("grounded") else 0.0


def appropriate_abstention(item: Dict, response: str, cfg: dict) -> bool:
    """True if, for an out-of-scope/unanswerable item, the model appropriately
    deferred rather than over-claiming. Used to upgrade unsafe_confidence_rate beyond
    the keyword heuristic."""
    user = (
        f"KIND:\n{item.get('kind')}\n\n"
        f"QUESTION:\n{item.get('question', '')}\n\n"
        f"POLICY:\n{item.get('reference', '')}\n\n"
        f"ANSWER:\n{response}"
    )
    raw = _call_judge(ABSTENTION_SYSTEM, user, cfg)
    parsed = json.loads(raw)
    return bool(parsed.get("appropriate_abstention"))
