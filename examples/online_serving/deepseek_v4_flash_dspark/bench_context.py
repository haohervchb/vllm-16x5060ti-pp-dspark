#!/usr/bin/env python3
"""Validate a long prompt and an optional cached-prefix extension."""

import json
import os
import time
import urllib.request

from transformers import AutoTokenizer


URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8099/v1/completions")
MODEL = os.environ.get("SERVED_MODEL_NAME", "vllm")
MODEL_PATH = os.environ["MODEL_PATH"]
TARGET_TOKENS = int(os.environ["TARGET_TOKENS"])
EXTEND_TOKENS = int(os.environ.get("EXTEND_TOKENS", "0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))

SEED_TEXT = (
    "Modern PCIe systems map device memory through bridge windows. A GPU "
    "runtime schedules kernels, transfers tensors, and maintains a paged "
    "key-value cache. Pipeline parallel inference assigns complete transformer "
    "layers to stages while tensor parallelism shards each layer. Prefix "
    "caching reuses immutable token blocks across requests. "
)


def run(prompt: list[int]) -> dict[str, object]:
    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(URL, payload, {"Content-Type": "application/json"})
    started = time.perf_counter()
    first_token = None
    usage = None
    chunks = 0
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            if event.get("choices"):
                first_token = first_token or time.perf_counter()
                chunks += 1
    finished = time.perf_counter()
    ttft = None if first_token is None else first_token - started
    return {
        "prompt_tokens": len(prompt),
        "usage": usage,
        "ttft_s": ttft,
        "effective_prompt_tok_s": None if ttft is None else len(prompt) / ttft,
        "elapsed_s": finished - started,
        "stream_chunks": chunks,
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    seed = tokenizer.encode(SEED_TEXT * 200, add_special_tokens=False)
    largest = max(TARGET_TOKENS, EXTEND_TOKENS)
    prompt = (seed * ((largest + len(seed) - 1) // len(seed)))[:largest]

    print(json.dumps({"cold": run(prompt[:TARGET_TOKENS])}, indent=2))
    if EXTEND_TOKENS:
        if EXTEND_TOKENS <= TARGET_TOKENS:
            raise SystemExit("EXTEND_TOKENS must be greater than TARGET_TOKENS")
        print(json.dumps({"cached_extension": run(prompt)}, indent=2))


if __name__ == "__main__":
    main()
