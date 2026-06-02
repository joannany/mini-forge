"""
dashboard/app.py

Streamlit dashboard for baseline metrics, tuned metrics, and the gate decision.
Reads results/latest.json and displays usefulness gain, regression flags, and
promotion status.

Run:  streamlit run dashboard/app.py
"""
import json
import os

import pandas as pd
import streamlit as st

RESULTS = os.environ.get("MINIFORGE_RESULTS", "results/latest.json")

LOWER_IS_BETTER = {"hallucination_rate", "over_refusal_rate", "unsafe_confidence_rate"}

st.set_page_config(page_title="Mini-Forge", layout="wide")
st.title("Mini-Forge — deployment-readiness")
st.caption("Did customization make the model more useful, without behavioral regressions?")

if not os.path.exists(RESULTS):
    st.warning(f"No results at {RESULTS}. Run `python -m eval.harness --config config.yaml`.")
    st.stop()

data = json.load(open(RESULTS))
base, tuned, gate = data["baseline"], data["tuned"], data["gate"]

# Headline: the gate decision.
if gate.get("smoke_test"):
    verdict = "SMOKE TEST"
    st.header("\u2139\ufe0f " + verdict)
    st.info("Fixture mode validates plumbing only. It is not a deployment gate.")
else:
    verdict = "PROMOTE" if gate["promote"] else "BLOCK"
    st.header(("\u2705 " if gate["promote"] else "\u26d4 ") + verdict)
    st.metric("Usefulness gain (tuned \u2212 baseline)", f"{gate['usefulness_gain']:+.3f}")
for r in gate["reasons"]:
    st.write("- " + r)

# Metric comparison table with regression direction made explicit.
st.subheader("Baseline vs tuned")
rows = []
for metric in sorted(set(base) | set(tuned)):
    b, t = base.get(metric), tuned.get(metric)
    if b is None or t is None:
        continue
    delta = t - b
    worse = (delta > 0) if metric in LOWER_IS_BETTER else (delta < 0)
    flag = "\u26a0\ufe0f regression" if (worse and metric in LOWER_IS_BETTER | {"format_adherence", "tool_use_reliability"}) else ""
    rows.append({"metric": metric, "baseline": round(b, 3),
                 "tuned": round(t, 3), "delta": round(delta, 3), "flag": flag})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if gate["regression_breaches"]:
    st.error("Regression breaches that blocked promotion: "
             + ", ".join(f"{k} ({v:+.3f})" for k, v in gate["regression_breaches"].items()))

st.caption(f"Generated {data.get('generated_at', '?')} \u00b7 source: {RESULTS}")
