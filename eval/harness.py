"""
eval/harness.py

Orchestrates the evaluation: run both suites on the baseline and the tuned model,
aggregate to a usefulness score, write a results JSON for the dashboard, and call the
gate. The per-metric logic lives in usefulness.py and regression.py.

Usage:
    python -m eval.harness --config config.yaml
"""
import argparse
import json
import os
from datetime import datetime

import yaml

from eval import usefulness, regression
from eval.gate import evaluate_gate
from eval.provider import generate_response


USEFULNESS_COMPONENTS = (
    "task_success",
    "rubric_score",
    "instruction_following",
    "factuality",
)


def load_eval_set(path):
    """Held-out eval set: one JSON object per line.
    Expected fields per item: id, question, reference, kind
    where kind in {"answerable", "out_of_scope", "benign_in_scope", "tool"}.
    """
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_suite(model_name, eval_items, cfg, model_role):
    """Run one model over the eval set and return a flat metric dict.

    Responses are produced through eval.provider.generate_response according to
    generation.mode. Fixture mode reads precomputed response fields; other modes call
    the configured model provider.
    """
    responses = [
        generate_response(model_name, item, cfg, model_role=model_role)
        for item in eval_items
    ]

    metrics = {}
    # ---- usefulness ----
    metrics["task_success"] = usefulness.task_success(eval_items, responses)
    metrics["rubric_score"] = usefulness.rubric_score(eval_items, responses, cfg)
    metrics["instruction_following"] = usefulness.instruction_following(eval_items, responses)
    metrics["factuality"] = usefulness.factuality(eval_items, responses, cfg)
    # Aggregate usefulness over available numeric components. Optional judge-backed
    # metrics can return no numeric value when a judge provider is unavailable.
    metrics["usefulness"] = mean_available(metrics, USEFULNESS_COMPONENTS)

    # ---- regression ----
    metrics["hallucination_rate"] = regression.hallucination_rate(eval_items, responses, cfg)
    metrics["over_refusal_rate"] = regression.over_refusal_rate(eval_items, responses)
    metrics["unsafe_confidence_rate"] = regression.unsafe_confidence_rate(eval_items, responses)
    metrics["format_adherence"] = regression.format_adherence(eval_items, responses)
    metrics["tool_use_reliability"] = regression.tool_use_reliability(eval_items, responses)
    return metrics


def mean_available(metrics, keys):
    values = [
        metrics[key]
        for key in keys
        if isinstance(metrics.get(key), (int, float))
    ]
    if not values:
        raise ValueError(f"no numeric usefulness components available: {keys}")
    return sum(values) / len(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    eval_items = load_eval_set(cfg["data"]["eval_set"])
    validate_fixture_gate_inputs(eval_items, cfg)

    baseline = run_suite(cfg["models"]["base_model"], eval_items, cfg, model_role="base")
    tuned = run_suite(cfg["models"]["tuned_model_path"], eval_items, cfg, model_role="tuned")

    smoke_test = is_fixture_smoke_test(cfg)
    if smoke_test:
        decision = None
        gate = {
            "promote": False,
            "usefulness_gain": None,
            "reasons": [
                "fixture smoke test only; no real baseline-vs-tuned model comparison",
                "generate real model responses with scripts/generate_prompt_baseline.py or scripts/generate_eval_responses.py, then set fixture_smoke_test=false",
            ],
            "regression_breaches": {},
            "smoke_test": True,
        }
    else:
        decision = evaluate_gate(baseline, tuned, cfg["gate"])
        gate = {
            "promote": decision.promote,
            "usefulness_gain": decision.usefulness_gain,
            "reasons": decision.reasons,
            "regression_breaches": decision.regression_breaches,
            "smoke_test": False,
        }

    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "generation_mode": (cfg.get("generation") or {}).get("mode", "fixture"),
        "baseline": baseline,
        "tuned": tuned,
        "gate": gate,
    }
    os.makedirs(cfg["results_dir"], exist_ok=True)
    out_path = os.path.join(cfg["results_dir"], "latest.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    verdict = "SMOKE" if smoke_test else ("PROMOTE" if decision.promote else "BLOCK")
    if smoke_test:
        print(f"[{verdict}] fixture plumbing check -> {out_path}")
    else:
        print(f"[{verdict}] usefulness {decision.usefulness_gain:+.3f} -> {out_path}")
    for r in gate["reasons"]:
        print("  -", r)


def is_fixture_smoke_test(cfg):
    generation_cfg = cfg.get("generation", {}) or {}
    return (
        generation_cfg.get("mode", "fixture") == "fixture"
        and generation_cfg.get("fixture_smoke_test", True)
    )


def validate_fixture_gate_inputs(eval_items, cfg):
    generation_cfg = cfg.get("generation", {}) or {}
    if generation_cfg.get("mode", "fixture") != "fixture":
        return
    if generation_cfg.get("fixture_smoke_test", True):
        return

    fields = generation_cfg.get("fixture_response_fields", {})
    base_field = fields.get("base", "base_response")
    tuned_field = fields.get("tuned", "tuned_response")
    missing = []
    fixture_sources = []
    for item in eval_items:
        if not item.get(base_field) or not item.get(tuned_field):
            missing.append(item.get("id", "?"))
            continue
        base_source = item.get(f"{base_field}_source") or item.get("base_response_source")
        tuned_source = item.get(f"{tuned_field}_source") or item.get("tuned_response_source")
        if base_source in (None, "", "gold_fixture") or tuned_source in (None, "", "gold_fixture"):
            fixture_sources.append(item.get("id", "?"))

    if missing:
        raise SystemExit(
            "ABORT: fixture gate requires non-empty real model responses. "
            f"Missing fields for eval ids: {missing[:10]}"
        )
    if fixture_sources:
        raise SystemExit(
            "ABORT: fixture_smoke_test=false requires model-generated response sources, "
            "not gold fixtures. Run scripts/generate_eval_responses.py first. "
            f"Fixture-only ids: {fixture_sources[:10]}"
        )


if __name__ == "__main__":
    main()
