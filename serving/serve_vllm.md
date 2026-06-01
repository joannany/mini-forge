# Serving — and how it scales to Mistral-Medium-3.5-128B

## MVP: serve the tuned small model

vLLM exposes an OpenAI-compatible server, so the dashboard, the judge, and any client
talk to it the same way they would talk to a hosted API.

```bash
pip install vllm
# merge the LoRA adapter into the base model first (or pass --enable-lora)
python -m vllm.entrypoints.openai.api_server \
  --model ./training/outputs/tuned-merged \
  --port 8000
# now: POST http://localhost:8000/v1/chat/completions
```

If GPU access is tight, it is legitimate to run this once, capture the request/response
and a screenshot, and document the rest — the workflow is the deliverable, not uptime.

## Scale target: Mistral-Medium-3.5-128B

Nothing in the data prep, eval suites, gate, or dashboard changes at 128B. Two things do:

1. **Infrastructure.** Medium 3.5 is a dense 128B model. The official vLLM example
   currently serves it with 8-way tensor parallelism:

   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model mistralai/Mistral-Medium-3.5-128B \
     --tensor-parallel-size 8
   ```

   Smaller GPU counts should be treated as a quantized/third-party deployment variant,
   not as the default claim for the official full model.

   (SGLang is an alternative engine; a NIM-style container is the enterprise path.)

2. **License.** Medium 3.5 ships under a modified MIT license that is permissive for
   most, but requires a separate commercial arrangement above an enterprise revenue
   threshold — i.e. exactly the large-enterprise / government buyers Forge targets.
   A deployment-readiness review for these customers must include this.

The point of the project: prove the gates on a 7–8B model, then show the same
gates and serving path scale to 128B. That is more credible than claiming to
fine-tune a 128B model locally.
