# Mini-Forge

A post-training **evaluation and deployment-readiness lab** for regulated enterprises:
customize a small open-weight model on domain data, then enforce model-quality
**gates** before anything is promoted to production.

> **Central product question**
> *Did customization make the model more useful for the target use case,
> without introducing behavioral regressions (hallucination, over-refusal,
> or unsafe confidence)?*

The project proves the full workflow on a small Mistral model and **documents how the
same gates and serving path scale to Mistral-Medium-3.5-128B** — which is more credible
than claiming to casually fine-tune a 128B model locally.

---

## Domain: regulated enterprise policy QA / compliance response generation

Take (public proxies for) internal policy documents → generate synthetic instruction
data → fine-tune a small model → evaluate whether the tuned model answers compliance
questions better, without becoming less safe or less reliable.

### What the fine-tune is (and is NOT) for

This is the single most important design decision, and the one most likely to be
challenged.

- **Fine-tune teaches *behavior*** — compliant answer format, citation/grounding
  discipline, calibrated refusal, tone, and safety posture.
- **RAG / context supplies *content*** — the current policy text. Policies change
  often; you do **not** retrain to absorb a policy edit.
- Mini-Forge therefore **isolates and measures the behavioral delta** from
  fine-tuning, holding content constant.

Fine-tuning a model to *memorize policy facts* is the canonical misuse of fine-tuning.
Mini-Forge deliberately does not do that.

### Data

Use **public regulatory text** as a stand-in for proprietary policy docs:
GDPR, HIPAA published rules, the EU AI Act, NIST frameworks, a public code of conduct
or employee handbook. In a real Forge deployment this corpus would be the customer's
internal documents. **Never use confidential or proprietary material.**

---

## Architecture

```
policy docs
   -> synthetic instruction-data generation        (data/generate_synthetic.py)
   -> validation / dedupe / PII scan / leakage check (data/prepare_data.py)
   -> SFT or LoRA job                                (training/train_lora.py)
   -> baseline vs tuned evaluation                   (eval/harness.py)
   -> deployment gate                                (eval/gate.py)
   -> vLLM serving                                   (serving/serve_vllm.md)
   -> monitoring / regression alerts                 (dashboard/app.py)
```

**Build vs borrow:** borrow the training (Unsloth) and serving (vLLM); **build the
evaluation, the gate, and the dashboard yourself** — that is the differentiator and
the part you must be able to defend.

---

## Evaluation design (the centerpiece)

Two suites plus a promotion gate.

**Usefulness** (`eval/usefulness.py`)
- task success on a domain question set
- rubric score via LLM-as-judge
- instruction / format following
- factuality / grounding against the source policy

**Regression** (`eval/regression.py`)
- hallucination rate
- over-refusal rate (benign, in-scope questions the model *should* answer)
- unsafe confidence — a calibration / abstention test: on unanswerable or
  out-of-scope questions, does the model over-claim or appropriately defer?
- format adherence (compliant output structure)
- tool-use reliability (if tool-calling is in scope)

**Gate** (`eval/gate.py`)
- promote the tuned model **only if** usefulness improves **and** every regression
  metric stays within its threshold. Thresholds live in `config.yaml`.

The over-refusal vs safety tension is a feature to surface, not hide: a model that
refuses everything is "safe" and useless. The dashboard shows both.

By default, factuality and hallucination use deterministic proxy checks so the project
runs without a judge key. `eval/judge.py` defines the LLM-as-judge interface and prompts,
but the provider call is intentionally unwired and fails loudly until configured; this
prevents a stub judge from silently corrupting grounding metrics.

---

## Scaling to Mistral-Medium-3.5-128B

Medium 3.5 is open-weight (modified MIT license, with a commercial exception above an
enterprise revenue threshold) and is served with vLLM or SGLang on multi-GPU
infrastructure. The official vLLM example currently uses `--tensor-parallel-size 8`;
smaller GPU counts may be possible only with quantized or third-party variants. The MVP
runs on a 7–8B model so iteration is fast and cheap; the *same* data prep, eval suites,
gate logic, and serving path apply unchanged at 128B. The main delta is infrastructure
and the enterprise license note — both documented in `serving/serve_vllm.md`, not
hand-waved.

---

## Quickstart: runnable MVP

```bash
pip install -r requirements.txt

# 1. generate train/eval data from local public policy docs
python -m data.generate_synthetic \
  --docs data/policy_docs \
  --out data/synthetic_train.jsonl \
  --eval-out data/eval_set.jsonl

# 2. validate schema, dedupe, PII, and train/eval leakage
python -m data.prepare_data \
  --train data/synthetic_train.jsonl \
  --eval data/eval_set.jsonl \
  --out data/synthetic_train.clean.jsonl

# 3. run fixture-mode plumbing smoke test
python -m eval.harness --config config.yaml

# 4. view smoke-test results
streamlit run dashboard/app.py
```

Fixture mode is intentionally included so the eval/dashboard plumbing is runnable on
a laptop. It is **not** a model-quality result and does **not** run the deployment
gate.

For real baseline-vs-tuned numbers, use the recommended Colab workflow: train the LoRA,
precompute base/tuned responses into `data/eval_set.jsonl`, then run the harness with
`fixture_smoke_test: false`. See [`docs/colab_workflow.md`](docs/colab_workflow.md).
Serving through vLLM or another OpenAI-compatible endpoint is also supported, but it is
not the easiest path inside a single Colab session.

## Real model paths

### Recommended Colab path: precompute real responses

For a single-GPU Colab/RunPod run, do **not** run a localhost vLLM server just to
evaluate. Train the LoRA, generate base/tuned responses in the notebook, write them
into `data/eval_set.jsonl`, then run the harness with `fixture_smoke_test: false`.

See [`docs/colab_workflow.md`](docs/colab_workflow.md).

### Path A: evaluate a served model

1. Serve a model with vLLM, SGLang, Mistral API, or any OpenAI-compatible endpoint.
2. Set:

```yaml
generation:
  mode: "openai_compatible"
  base_url: "http://localhost:8000/v1"
  api_key_env: "OPENAI_API_KEY"
```

3. Disable fixture smoke mode and run the real gate:

```bash
# config.yaml:
# generation.mode: "openai_compatible"
# generation.fixture_smoke_test: false
python -m eval.harness --config config.yaml
```

### Path B: train a real LoRA adapter

Run this in a CUDA environment with Unsloth/TRL installed:

```bash
python -m training.train_lora \
  --train data/synthetic_train.clean.jsonl \
  --base-model unsloth/mistral-7b-instruct-v0.3-bnb-4bit \
  --out training/outputs/tuned-lora \
  --max-steps 80
```

Then serve the adapter or a merged model via vLLM and rerun the same evaluation gate.

## Layout

| Path | Purpose | Status |
|---|---|---|
| `data/generate_synthetic.py` | synthetic instruction data from policy docs | working |
| `data/prepare_data.py` | validation, dedupe, PII scan, leakage check | working |
| `training/train_lora.py` | QLoRA/SFT training entry point for GPU envs | working |
| `scripts/generate_eval_responses.py` | Colab response precompute for real fixture gates | working |
| `docs/colab_workflow.md` | reproducible Colab workflow for real numbers | doc |
| `eval/provider.py` | fixture / OpenAI-compatible / transformers generation | working |
| `eval/harness.py` | orchestrates suites, writes results, calls gate | working |
| `eval/usefulness.py` | usefulness metrics | working |
| `eval/regression.py` | regression metrics | working |
| `eval/judge.py` | optional judge prompts; proxy metrics run by default, provider call unwired | optional |
| `eval/gate.py` | promotion decision | working |
| `dashboard/app.py` | baseline vs tuned vs deployed dashboard | working shell |
| `serving/serve_vllm.md` | how to serve, and how it scales to 128B | doc |
