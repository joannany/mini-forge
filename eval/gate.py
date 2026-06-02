"""
eval/gate.py

The deployment gate: the single decision Mini-Forge exists to make.

Promote the tuned model to production ONLY if:
  (1) usefulness improved by at least `min_usefulness_gain`, AND
  (2) every regression metric stayed within its allowed delta vs the baseline.

Upstream evaluation code produces the metric dictionaries consumed by this module.
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GateDecision:
    promote: bool
    usefulness_gain: float
    reasons: List[str] = field(default_factory=list)
    regression_breaches: Dict[str, float] = field(default_factory=dict)


# Metrics where HIGHER is better (a drop below baseline is a regression).
HIGHER_IS_BETTER = {"format_adherence", "tool_use_reliability"}
# Metrics where LOWER is better (a rise above baseline is a regression).
LOWER_IS_BETTER = {"hallucination_rate", "over_refusal_rate", "unsafe_confidence_rate"}


def evaluate_gate(baseline: Dict[str, float],
                  tuned: Dict[str, float],
                  gate_cfg: dict) -> GateDecision:
    """
    baseline / tuned: flat dicts of metric_name -> score, each containing a
        "usefulness" aggregate plus the regression metrics named in config.
    gate_cfg: the `gate:` block from config.yaml.
    """
    reasons: List[str] = []
    breaches: Dict[str, float] = {}

    # (1) usefulness must improve enough
    usefulness_gain = tuned["usefulness"] - baseline["usefulness"]
    min_gain = gate_cfg["min_usefulness_gain"]
    usefulness_ok = usefulness_gain >= min_gain
    if not usefulness_ok:
        reasons.append(
            f"usefulness gain {usefulness_gain:+.3f} below required {min_gain:+.3f}"
        )

    # (2) no regression may exceed its allowed delta
    for metric, allowed in gate_cfg["max_regression_deltas"].items():
        delta = tuned.get(metric, 0.0) - baseline.get(metric, 0.0)
        if metric in LOWER_IS_BETTER:
            # delta > allowed means it got worse beyond tolerance
            if delta > allowed:
                breaches[metric] = delta
                reasons.append(f"{metric} rose {delta:+.3f} (allowed {allowed:+.3f})")
        elif metric in HIGHER_IS_BETTER:
            # delta < allowed (allowed is <= 0) means it dropped too far
            if delta < allowed:
                breaches[metric] = delta
                reasons.append(f"{metric} dropped {delta:+.3f} (allowed {allowed:+.3f})")

    promote = usefulness_ok and not breaches
    if promote:
        reasons.append("all gates passed")
    return GateDecision(
        promote=promote,
        usefulness_gain=usefulness_gain,
        reasons=reasons,
        regression_breaches=breaches,
    )
