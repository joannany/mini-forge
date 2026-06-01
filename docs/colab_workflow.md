# Colab Workflow: Real Baseline vs Tuned Gate

This is the recommended path for producing real Mini-Forge numbers on a single Colab
GPU. It does **not** run a local vLLM server. Instead, it generates base/tuned outputs
inside the notebook, writes them into `data/eval_set.jsonl`, then runs the same harness
in fixture mode with the smoke-test guard disabled.

## 1. Clone the repo

```python
!git clone https://github.com/joannany/mini-forge.git
%cd mini-forge
```

If the repo is private, use a GitHub token or Colab's GitHub integration.

## 2. Install dependencies

```python
!pip install -q unsloth trl datasets pyyaml
```

If Unsloth changes its install instructions, use the current official Unsloth Colab
setup and keep the remaining commands the same.

## 3. Generate and validate data

```python
!python -m data.generate_synthetic \
  --docs data/policy_docs \
  --out data/synthetic_train.jsonl \
  --eval-out data/eval_set.jsonl

!python -m data.prepare_data \
  --train data/synthetic_train.jsonl \
  --eval data/eval_set.jsonl \
  --out data/synthetic_train.clean.jsonl
```

## 4. Train a LoRA adapter

```python
!python -m training.train_lora \
  --train data/synthetic_train.clean.jsonl \
  --base-model unsloth/Ministral-8B-Instruct-2410 \
  --out training/outputs/tuned-lora \
  --max-steps 80
```

If adapter loading fails later, merge/save the model using Unsloth's current notebook
pattern, then pass the merged model directory as `--tuned-model`.

## 5. Generate real eval responses

```python
!python -m scripts.generate_eval_responses \
  --eval data/eval_set.jsonl \
  --base-model unsloth/Ministral-8B-Instruct-2410 \
  --tuned-model training/outputs/tuned-lora
```

This writes:

- `base_response`
- `tuned_response`
- `base_response_source`
- `tuned_response_source`

The source fields are required before the harness will run a non-smoke fixture gate.

## 6. Run the real gate

```python
!python - <<'PY'
import yaml
path = "config.yaml"
cfg = yaml.safe_load(open(path))
cfg["generation"]["mode"] = "fixture"
cfg["generation"]["fixture_smoke_test"] = False
yaml.safe_dump(cfg, open(path, "w"))
PY

!python -m eval.harness --config config.yaml
```

Now the gate is real because the fixture fields contain model-generated outputs, not
gold answers or empty strings.

## Optional: serving artifact

For a portfolio screenshot, serve the model separately with vLLM or SGLang and hit a
few `/v1/chat/completions` requests. Keep that serving demo separate from the eval loop;
the eval numbers above are easier to reproduce and less fragile in Colab.
