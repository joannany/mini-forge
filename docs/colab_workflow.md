# Colab Workflow: Real Before/After Gate

This is the recommended path for producing real Mini-Forge numbers on a single Colab
GPU. It does **not** run a local vLLM server. It generates outputs inside the notebook,
writes them into `data/eval_set.jsonl`, then runs the same harness in fixture mode with
the smoke-test guard disabled.

Use the prompt-intervention path first. It produces real before/after gate numbers
without LoRA training. The source-tag and gate contract for this path is validated in
the repo; the model generation step runs in Colab. The LoRA path is implemented but
still needs GPU validation end to end.

## 1. Clone the repo

```python
!git clone https://github.com/joannany/mini-forge.git
%cd mini-forge
```

If the repo is private, use a GitHub token or Colab's GitHub integration.

## 2. Install dependencies

```python
!pip install -q unsloth datasets transformers accelerate pyyaml
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

## 4. Fast path: prompt intervention

This measures a behavioral intervention through the gate:

- baseline = same model with a bare prompt
- tuned = same model with Mini-Forge's compliant system prompt

It is not a fine-tune, and should be described that way.

```python
!python -m scripts.generate_prompt_baseline \
  --eval data/eval_set.jsonl \
  --model unsloth/mistral-7b-instruct-v0.3-bnb-4bit
```

Then run the real gate:

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

## 5. Optional full path: train a LoRA adapter

This path is implemented, but it has not yet been validated end to end on a Colab GPU.
Use it as the fuller LoRA version after the prompt-intervention result is captured.

```python
!python -m training.train_lora \
  --train data/synthetic_train.clean.jsonl \
  --base-model unsloth/mistral-7b-instruct-v0.3-bnb-4bit \
  --out training/outputs/tuned-lora \
  --max-steps 80
```

The script defaults to `--precision fp16`, which is required on free Colab T4 GPUs.
Use `--precision bf16` only on Ampere+ GPUs such as A100/L4/H100.
It pre-tokenizes examples and uses `transformers.Trainer` directly to avoid TRL's
dataset multiprocessing path.

If adapter loading fails later, merge/save the model using Unsloth's current notebook
pattern, then pass the merged model directory as `--tuned-model`.

## 6. Generate LoRA eval responses

```python
!python -m scripts.generate_eval_responses \
  --eval data/eval_set.jsonl \
  --base-model unsloth/mistral-7b-instruct-v0.3-bnb-4bit \
  --tuned-model training/outputs/tuned-lora
```

This writes:

- `base_response`
- `tuned_response`
- `base_response_source`
- `tuned_response_source`

The source fields are required before the harness will run a non-smoke fixture gate.

## 7. Run the real LoRA gate

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

## Optional: serving artifact

For a portfolio screenshot, serve the model separately with vLLM or SGLang and hit a
few `/v1/chat/completions` requests. Keep that serving demo separate from the eval loop;
the eval numbers above are easier to reproduce and less fragile in Colab.
