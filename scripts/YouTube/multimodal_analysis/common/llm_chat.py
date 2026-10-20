"""Unified chat-completion interface across the three backends stage 8/9's
ensemble models use (see config.ENSEMBLE_MODELS' "backend" field):

- "vllm_local": a small (~8-12B) model loaded once into a local vLLM engine
  on the job's own GPU. Calls are batched through vLLM's offline-inference
  API (llm.chat() given a list of conversations at once) so its scheduler
  can pack many requests together.
- "aitta": a big (~27-70B) model with no local GPU -- a network call to CSC
  LUMI's hosted-inference API (config.AITTA_BASE_URL). Requests go out from
  a worker-thread pool (overlaps network round-trip latency) with retry/
  backoff on the per-minute rate limit (HTTP 429).
- "alibaba": same shape as "aitta", but against Alibaba Cloud Model
  Studio's DashScope international endpoint (config.ALIBABA_BASE_URL).
  Used for qwen36_27b: Aitta hosts this model too but doesn't forward
  enable_thinking, so its hidden <think> chain-of-thought silently eats the
  completion budget there (see config.py's ENSEMBLE_MODELS comment).
  DashScope honors extra_body={"enable_thinking": False} correctly.

All three are exposed through the same chat_batch(messages_list, model_entry,
max_tokens) -> list[str] call site, so stage 8/9 never need to know which
backend a given config.ENSEMBLE_MODELS entry uses.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from . import config

load_dotenv()

# Without this, huggingface_hub falls back to ~/.cache/huggingface -- on
# Roihu that's the /users home filesystem, a 15G quota shared with
# everything else, not the project's /scratch allocation. An 8B model in
# bf16 is ~16G on its own, so it blows that quota immediately. Set before
# anything that might import transformers/vllm, since HF_HOME governs their
# downloads too, not just this module's.
Path(config.CACHE_DIR).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(config.CACHE_DIR))

_vllm_engines: dict[str, object] = {}  # hf_name -> vllm.LLM, populated lazily


def _get_vllm_engine(hf_name: str):
    if hf_name not in _vllm_engines:
        from vllm import LLM  # imported lazily: keeps this module importable
                               # on the Aitta-only, no-GPU adjudication path
        print(f"[llm_chat] loading {hf_name} into a local vLLM engine...")
        _vllm_engines[hf_name] = LLM(model=hf_name, trust_remote_code=True, download_dir=str(config.CACHE_DIR))
    return _vllm_engines[hf_name]


def _chat_batch_vllm_local(messages_list: list[list[dict]], hf_name: str, max_tokens: int) -> list[str]:
    from vllm import SamplingParams
    llm = _get_vllm_engine(hf_name)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.chat(messages_list, sampling_params=params, use_tqdm=False)
    return [out.outputs[0].text for out in outputs]


_aitta_client = None
_aitta_client_lock = threading.Lock()


def _get_aitta_client():
    # Double-checked locking: called from a pool of worker threads, so the
    # first-call-wins init needs to be atomic.
    global _aitta_client
    if _aitta_client is None:
        with _aitta_client_lock:
            if _aitta_client is None:
                import openai
                token = os.environ["AITTA_API_TOKEN"]
                _aitta_client = openai.OpenAI(api_key=token, base_url=config.AITTA_BASE_URL)
    return _aitta_client


def _chat_one_aitta(messages: list[dict], hf_name: str, max_tokens: int) -> str:
    """Never raises: nothing gets written to disk until the whole batch
    returns (see 08_ensemble_label_chunks.py), so one permanently-failing
    request must not lose every already-completed classification with it.
    A 429 is retried with backoff; any other failure is logged and degrades
    to an empty response for just that one item -- io_utils.parse_label_list("")
    -> [], same as any other unparseable response."""
    client = _get_aitta_client()
    for attempt in range(config.AITTA_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=hf_name, messages=messages, max_tokens=max_tokens, temperature=0.0,
            )
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                # Hit max_tokens before finishing -- for a reasoning model
                # spending its budget on hidden chain-of-thought, this can
                # mean empty/truncated content with no exception raised.
                print(f"[llm_chat] Aitta response from {hf_name} hit max_tokens={max_tokens} "
                      f"(finish_reason=length, content length={len(choice.message.content or '')})")
            return choice.message.content or ""
        except Exception as e:
            is_rate_limited = "429" in str(e)
            if is_rate_limited and attempt < config.AITTA_MAX_RETRIES - 1:
                print(f"[llm_chat] Aitta rate-limited (attempt {attempt + 1}/{config.AITTA_MAX_RETRIES}), "
                      f"waiting {config.AITTA_RATE_LIMIT_BACKOFF_SECONDS}s...")
                time.sleep(config.AITTA_RATE_LIMIT_BACKOFF_SECONDS)
                continue
            print(f"[llm_chat] Aitta request to {hf_name} failed, skipping this item: {e}")
            return ""
    return ""


def _chat_batch_aitta(messages_list: list[list[dict]], hf_name: str, max_tokens: int) -> list[str]:
    # Requests go out concurrently (config.AITTA_MAX_CONCURRENT_REQUESTS at a
    # time) to overlap network round-trip latency instead of serializing it.
    # Each request still retries/backs off on 429 independently.
    results: list[str] = [""] * len(messages_list)
    start = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=config.AITTA_MAX_CONCURRENT_REQUESTS) as pool:
        future_to_index = {
            pool.submit(_chat_one_aitta, messages, hf_name, max_tokens): i
            for i, messages in enumerate(messages_list)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            results[i] = future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(messages_list):
                elapsed = time.time() - start
                print(f"[llm_chat] {hf_name}: {completed}/{len(messages_list)} requests done "
                      f"({elapsed:.0f}s elapsed, {elapsed / completed:.2f}s/req)", flush=True)
    return results


_alibaba_client = None
_alibaba_client_lock = threading.Lock()


def _get_alibaba_client():
    global _alibaba_client
    if _alibaba_client is None:
        with _alibaba_client_lock:
            if _alibaba_client is None:
                import openai
                key = os.environ["QWEN_API_KEY"]
                _alibaba_client = openai.OpenAI(api_key=key, base_url=config.ALIBABA_BASE_URL)
    return _alibaba_client


def _chat_one_alibaba(messages: list[dict], model_id: str, max_tokens: int) -> str:
    """Mirrors _chat_one_aitta (never raises, degrades a single failing item
    to ""), with enable_thinking forced off via extra_body -- see this
    module's docstring for why."""
    client = _get_alibaba_client()
    for attempt in range(config.ALIBABA_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model_id, messages=messages, max_tokens=max_tokens, temperature=0.0,
                extra_body={"enable_thinking": False},
            )
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                print(f"[llm_chat] Alibaba response from {model_id} hit max_tokens={max_tokens} "
                      f"(finish_reason=length, content length={len(choice.message.content or '')})")
            return choice.message.content or ""
        except Exception as e:
            is_rate_limited = "429" in str(e)
            if is_rate_limited and attempt < config.ALIBABA_MAX_RETRIES - 1:
                print(f"[llm_chat] Alibaba rate-limited (attempt {attempt + 1}/{config.ALIBABA_MAX_RETRIES}), "
                      f"waiting {config.ALIBABA_RATE_LIMIT_BACKOFF_SECONDS}s...")
                time.sleep(config.ALIBABA_RATE_LIMIT_BACKOFF_SECONDS)
                continue
            print(f"[llm_chat] Alibaba request to {model_id} failed, skipping this item: {e}")
            return ""
    return ""


def _chat_batch_alibaba(messages_list: list[list[dict]], model_id: str, max_tokens: int) -> list[str]:
    """Mirrors _chat_batch_aitta's worker-thread-pool shape."""
    results: list[str] = [""] * len(messages_list)
    start = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=config.ALIBABA_MAX_CONCURRENT_REQUESTS) as pool:
        future_to_index = {
            pool.submit(_chat_one_alibaba, messages, model_id, max_tokens): i
            for i, messages in enumerate(messages_list)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            results[i] = future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(messages_list):
                elapsed = time.time() - start
                print(f"[llm_chat] {model_id}: {completed}/{len(messages_list)} requests done "
                      f"({elapsed:.0f}s elapsed, {elapsed / completed:.2f}s/req)", flush=True)
    return results


def chat_batch(messages_list: list[list[dict]], model_entry: dict, max_tokens: int) -> list[str]:
    """messages_list: one OpenAI-style conversation (list of {"role",
    "content"} dicts) per item. Returns the generated text for each, same
    order. model_entry is one of config.ENSEMBLE_MODELS' dicts."""
    if not messages_list:
        return []
    backend = model_entry["backend"]
    if backend == "vllm_local":
        return _chat_batch_vllm_local(messages_list, model_entry["hf_name"], max_tokens)
    elif backend == "aitta":
        return _chat_batch_aitta(messages_list, model_entry["hf_name"], max_tokens)
    elif backend == "alibaba":
        return _chat_batch_alibaba(messages_list, model_entry["hf_name"], max_tokens)
    raise ValueError(f"Unknown backend {backend!r} for model {model_entry['key']!r}")


def chat(messages: list[dict], model_entry: dict, max_tokens: int) -> str:
    """Single-conversation convenience wrapper around chat_batch(), for
    call sites (e.g. stage 9's low-volume adjudication) that don't have a
    batch of prompts to build up front."""
    return chat_batch([messages], model_entry, max_tokens)[0]
