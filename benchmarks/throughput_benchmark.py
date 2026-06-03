#!/usr/bin/env python3
"""Throughput benchmark for vLLM ROCm."""

import argparse
import time
from typing import List

import torch

from src.engine import InferenceEngine


def benchmark_throughput(
    engine: InferenceEngine,
    num_prompts: int = 1000,
    input_len: int = 1024,
    output_len: int = 256,
    batch_sizes: List[int] = None,
) -> dict:
    """Measure throughput at various batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    results = []
    for bs in batch_sizes:
        print(f"\nBatch size: {bs}")

        # Create synthetic prompts
        prompts = []
        for _ in range(min(num_prompts, bs * 10)):
            text = "Explain the theory of general relativity in detail." + " " * input_len
            prompts.append(text)

        # Warmup
        for p in prompts[:2]:
            engine.generate(p, max_tokens=10)

        # Benchmark
        total_tokens = 0
        start = time.perf_counter()

        completed = 0
        for i in range(0, min(num_prompts, bs * 10), bs):
            batch = prompts[i:i + bs]
            for prompt in batch:
                output = engine.generate(prompt, max_tokens=output_len)
                total_tokens += output["completion_tokens"]
                completed += 1

        elapsed = time.perf_counter() - start
        tokens_per_sec = total_tokens / elapsed
        requests_per_sec = completed / elapsed
        latency_ms = (elapsed / completed) * 1000

        result = {
            "batch_size": bs,
            "completed": completed,
            "total_tokens": total_tokens,
            "elapsed_sec": round(elapsed, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "requests_per_sec": round(requests_per_sec, 2),
            "avg_latency_ms": round(latency_ms, 2),
        }
        results.append(result)
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec, "
              f"{requests_per_sec:.0f} req/sec, latency={latency_ms:.1f}ms")

    return {"results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=256)
    args = parser.parse_args()

    engine = InferenceEngine(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )

    results = benchmark_throughput(
        engine=engine,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Throughput Benchmark Results")
    print("=" * 60)
    for r in results["results"]:
        print(f"  BS={r['batch_size']:3d}: {r['tokens_per_sec']:>8.0f} tok/s  "
              f"{r['requests_per_sec']:>6.0f} req/s  "
              f"latency={r['avg_latency_ms']:>6.1f}ms")


if __name__ == "__main__":
    main()
