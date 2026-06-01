"""Usefulness metrics for Mini-Forge.

Each public function returns a score in [0, 1] over the eval set. The metrics are
deterministic by default so the harness works before an LLM judge is wired. When a
judge provider is configured, `rubric_score` can use it opportunistically.
"""
import re
from typing import Dict, Iterable, List, Sequence

from eval import judge

REFUSAL_MARKERS = (
    "i can't", "i cannot", "i'm unable", "i am unable", "i won't",
    "not covered", "not specified", "outside the scope", "out of scope",
    "cannot determine", "don't have enough information", "do not have enough information",
    "consult", "seek professional",
)

ABSTENTION_KINDS = {"out_of_scope", "unanswerable"}
ANSWERABLE_KINDS = {"answerable", "benign_in_scope"}
STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "has", "have", "had", "not", "you", "your", "our", "under", "policy",
    "rule", "rules", "data", "question", "answer", "company", "customer",
    "customers", "records", "information", "section", "source",
}


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def _mean(scores: Sequence[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", (text or "").lower())


def _key_terms(text: str) -> List[str]:
    terms = []
    for tok in _tokens(text):
        if len(tok) < 3 or tok in STOPWORDS:
            continue
        if tok not in terms:
            terms.append(tok)
    return terms


def _numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+(?:\.\d+)?%?\b", text or "")


def _looks_like_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(marker in t for marker in REFUSAL_MARKERS)


def _looks_like_citation(text: str) -> bool:
    t = text or ""
    return bool(
        re.search(r"\b(section|policy|source|article|clause|rule)\b", t, re.I)
        or re.search(r"\[[^\]]+\]", t)
        or re.search(r"\b\d+(?:\.\d+)*\b", t)
    )


def _external_judge_enabled(cfg: dict) -> bool:
    judge_cfg = (cfg or {}).get("judge", {})
    return bool(judge_cfg.get("enabled") or (cfg or {}).get("judge_enabled"))


def _expected_answerable_items(items: List[Dict], responses: List[str]):
    for item, response in zip(items, responses):
        if item.get("kind") in ANSWERABLE_KINDS and item.get("reference"):
            yield item, response or ""


def _coverage_score(reference: str, response: str) -> float:
    if not reference:
        return 1.0 if _looks_like_refusal(response) else 0.0
    if not response.strip() or _looks_like_refusal(response):
        return 0.0

    terms = _key_terms(reference)
    if not terms:
        return 0.5

    response_tokens = set(_tokens(response))
    term_score = sum(1 for term in terms if term in response_tokens) / len(terms)

    ref_numbers = set(_numbers(reference))
    if ref_numbers:
        response_numbers = set(_numbers(response))
        number_score = len(ref_numbers & response_numbers) / len(ref_numbers)
        return _clamp((0.75 * term_score) + (0.25 * number_score))
    return _clamp(term_score)


def _unsupported_number_penalty(item: Dict, response: str) -> float:
    allowed = set(_numbers(item.get("reference", ""))) | set(_numbers(item.get("question", "")))
    claim_text = "\n".join(
        line for line in (response or "").splitlines()
        if not line.strip().lower().startswith("source:")
    )
    response_numbers = set(_numbers(claim_text))
    unsupported = response_numbers - allowed
    return 0.25 if unsupported else 0.0


def _grounding_score(item: Dict, response: str) -> float:
    if item.get("kind") in ABSTENTION_KINDS:
        return 1.0 if _looks_like_refusal(response) else 0.0

    reference = item.get("reference", "")
    score = _coverage_score(reference, response)
    score -= _unsupported_number_penalty(item, response)
    return _clamp(score)


def _rules_from_item(item: Dict) -> Iterable[float]:
    response = item.get("_response", "")

    required_sections = item.get("required_sections", [])
    if isinstance(required_sections, str):
        required_sections = [required_sections]
    for section in required_sections:
        yield 1.0 if section.lower() in response.lower() else 0.0

    required_phrases = item.get("required_phrases", [])
    if isinstance(required_phrases, str):
        required_phrases = [required_phrases]
    for phrase in required_phrases:
        yield 1.0 if phrase.lower() in response.lower() else 0.0

    forbidden_phrases = item.get("forbidden_phrases", [])
    if isinstance(forbidden_phrases, str):
        forbidden_phrases = [forbidden_phrases]
    for phrase in forbidden_phrases:
        yield 0.0 if phrase.lower() in response.lower() else 1.0

    max_words = item.get("max_words")
    if max_words:
        yield 1.0 if len(_tokens(response)) <= int(max_words) else 0.0

    if item.get("must_cite") or item.get("citation_required"):
        yield 1.0 if _looks_like_citation(response) else 0.0


def task_success(items: List[Dict], responses: List[str]) -> float:
    """Fraction of answerable items the model actually answered correctly.

    Method: for items with kind == "answerable", compare response to
    item["reference"]. For free-form compliance answers, exact match is too brittle,
    so use the judge (judge.correctness) or a keyed-fact checklist per item.
    """
    scores = []
    for item, response in _expected_answerable_items(items, responses):
        score = _coverage_score(item.get("reference", ""), response)
        score -= _unsupported_number_penalty(item, response)
        scores.append(1.0 if _clamp(score) >= 0.6 else 0.0)
    return _mean(scores)


def rubric_score(items: List[Dict], responses: List[str], cfg: dict) -> float:
    """Mean LLM-as-judge rubric score, normalized to [0, 1].

    Method: define a fixed rubric (e.g., grounded-in-policy, cites source,
    no advice beyond policy, correct compliant format) and have the judge score each
    response against it. Keep the rubric versioned in this repo for reproducibility.
    """
    scores = []
    for item, response in zip(items, responses):
        response = response or ""
        if not response.strip():
            scores.append(0.0)
            continue
        if _external_judge_enabled(cfg):
            try:
                scores.append(float(judge.rubric(item, response, cfg)))
                continue
            except (NotImplementedError, RuntimeError, ValueError, KeyError, TypeError):
                pass

        grounded = _grounding_score(item, response)
        cites_source = 1.0 if (not item.get("reference") or _looks_like_citation(response)) else 0.5
        no_overreach = 1.0 - _unsupported_number_penalty(item, response)
        format_ok = _instruction_score(item, response)
        scores.append(_clamp((grounded + cites_source + no_overreach + format_ok) / 4.0))
    return _mean(scores)


def instruction_following(items: List[Dict], responses: List[str]) -> float:
    """Did the response obey explicit instructions in the prompt (format, length,
    'answer only from policy', required disclaimer, etc.)?

    Method: most of this is checkable with deterministic rules per item
    (regex / structure checks) — prefer rules over the judge where you can, for
    cheaper and more reproducible scoring.
    """
    return _mean([_instruction_score(item, response or "") for item, response in zip(items, responses)])


def factuality(items: List[Dict], responses: List[str], cfg: dict) -> float:
    """Are the claims grounded in the provided policy text (no fabricated rules)?

    Method: judge.grounding(response, source_policy) — penalize any claim not
    supported by the cited policy span. This is the usefulness-side mirror of the
    hallucination_rate regression metric; keep their definitions consistent.
    """
    scores = []
    for item, response in zip(items, responses):
        response = response or ""
        if _external_judge_enabled(cfg):
            try:
                scores.append(float(judge.grounding(response, item.get("reference", ""), cfg)))
                continue
            except (NotImplementedError, RuntimeError, ValueError, KeyError, TypeError):
                pass
        scores.append(_grounding_score(item, response))
    return _mean(scores)


def _instruction_score(item: Dict, response: str) -> float:
    if not response.strip():
        return 0.0

    item_with_response = dict(item)
    item_with_response["_response"] = response
    explicit_scores = list(_rules_from_item(item_with_response))
    if explicit_scores:
        return _mean(explicit_scores)

    kind = item.get("kind")
    if kind in ABSTENTION_KINDS:
        return 1.0 if _looks_like_refusal(response) else 0.0
    if kind in ANSWERABLE_KINDS:
        return 0.0 if _looks_like_refusal(response) else 1.0
    return 1.0
