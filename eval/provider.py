"""Model generation providers for Mini-Forge.

Provider modes:
- fixture: read precomputed responses from eval_set.jsonl.
- openai_compatible: call a vLLM/Mistral/OpenAI-compatible chat endpoint.
- transformers: run a local Hugging Face causal LM pipeline.

The harness uses the same interface for baseline and tuned models, so the eval/gate
logic is independent from serving infrastructure.
"""
import json
import os
import urllib.error
import urllib.request
from typing import Dict, List


SYSTEM_PROMPT = """You are a compliance assistant for regulated enterprise policy QA.
Answer only from the provided policy excerpt. If the excerpt does not answer the
question, say that it is not covered by the provided policy and do not invent a rule.
Use a concise structure:
Answer: ...
Source: ...
Risk note: ...
"""


def response_field_for(model_role: str, cfg: dict) -> str:
    fields = (cfg.get("generation", {}) or {}).get("fixture_response_fields", {})
    return fields.get(model_role, "response")


def build_messages(item: Dict) -> List[Dict[str, str]]:
    policy = item.get("reference") or "No applicable policy excerpt was provided."
    user = (
        f"POLICY EXCERPT:\n{policy}\n\n"
        f"QUESTION:\n{item.get('question', '')}\n\n"
        "Return the compliant answer."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_response(model_name: str, item: Dict, cfg: dict, model_role: str = "base") -> str:
    generation_cfg = cfg.get("generation", {}) or {}
    mode = generation_cfg.get("mode", "fixture")

    if mode == "fixture":
        return item.get(response_field_for(model_role, cfg), item.get("response", ""))
    if mode == "openai_compatible":
        return _openai_compatible_generate(model_name, item, cfg)
    if mode == "transformers":
        return _transformers_generate(model_name, item, cfg)
    raise ValueError(f"unknown generation.mode: {mode}")


def _openai_compatible_generate(model_name: str, item: Dict, cfg: dict) -> str:
    generation_cfg = cfg.get("generation", {}) or {}
    base_url = generation_cfg.get("base_url", "http://localhost:8000/v1").rstrip("/")
    api_key = os.environ.get(generation_cfg.get("api_key_env", "OPENAI_API_KEY"), "EMPTY")
    payload = {
        "model": model_name,
        "messages": build_messages(item),
        "temperature": generation_cfg.get("temperature", 0.0),
        "max_tokens": generation_cfg.get("max_tokens", 512),
    }
    if generation_cfg.get("reasoning_effort"):
        payload["reasoning_effort"] = generation_cfg["reasoning_effort"]

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=generation_cfg.get("timeout_seconds", 120)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"generation HTTP {exc.code}: {body}") from exc
    return data["choices"][0]["message"].get("content") or ""


def _transformers_generate(model_name: str, item: Dict, cfg: dict) -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "generation.mode=transformers requires torch and transformers. "
            "Install them or use generation.mode=openai_compatible."
        ) from exc

    generation_cfg = cfg.get("generation", {}) or {}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=generation_cfg.get("device_map", "auto"),
    )
    messages = build_messages(item)
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=generation_cfg.get("max_tokens", 512),
        do_sample=generation_cfg.get("temperature", 0.0) > 0,
        temperature=max(generation_cfg.get("temperature", 0.0), 1e-5),
    )
    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
