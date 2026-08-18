#!/usr/bin/env python3
"""Measure DSpark decode across content with different predictability."""

import json
import os
import time
import urllib.request

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8099/v1/completions")
MODEL = os.environ.get("SERVED_MODEL_NAME", "deepseek-ai/DeepSeek-V4-Flash-0731")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "400"))

PROMPTS = {
    "technical": (
        "Explain in careful detail how a modern operating system kernel implements "
        "virtual memory: page tables, the TLB, page faults, demand paging, "
        "copy-on-write, and swapping."
    ),
    "open-prose": (
        "Write an original short story about a lighthouse keeper who discovers "
        "something unexpected washed up on the shore one winter morning."
    ),
    "code": (
        "Write a complete Python implementation of a red-black tree with insert, "
        "delete, and search, including the rebalancing cases, with comments."
    ),
}


def request(prompt: str) -> tuple[int, float]:
    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(URL, payload, {"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as response:
        result = json.load(response)
    return result["usage"]["completion_tokens"], time.perf_counter() - started


def main() -> None:
    total_tokens = 0
    total_seconds = 0.0
    print(f"{'workload':>12} {'tokens':>7} {'seconds':>9} {'tok/s':>8}")
    for name, prompt in PROMPTS.items():
        tokens, seconds = request(prompt)
        total_tokens += tokens
        total_seconds += seconds
        print(f"{name:>12} {tokens:>7} {seconds:>9.3f} {tokens / seconds:>8.2f}")
    print(
        f"{'aggregate':>12} {total_tokens:>7} {total_seconds:>9.3f} "
        f"{total_tokens / total_seconds:>8.2f}"
    )


if __name__ == "__main__":
    main()
