#!/usr/bin/env python3
"""Latency benchmark for vLLM ROCm."""

import argparse
import statistics
import time
from typing import List

from src.engine import InferenceEngine


def benchmark_latency(
    engine: InferenceEngine,
    num_requests: int = 100,
    input_len: int = 128,
    output_len: int = 128,
) -> dict:
    """Measure end-to-end latency statistics."""
    prompt = "What is deep learning?" + " " * input_len

    # Warmup
    for _ in range(10):
        engine.generate(prompt, max_tokens=10)

    latencies = []
    token_latencies = []

    for i in range(num_requests):
        start = time.perf_counter()
        output = engine.generate(prompt, max_tokens=output_len)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        per_token_ms = latency_ms / max(output["completion_tokens"], 1)

        latencies.append(latency_ms)
        token_latencies.append(per_token_ms)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{num_requests}] p50={statistics.median(latencies):.1f}ms")

    results = {
        "num_requests": num_requests,
        "latency_p50_ms": round(statistics.median(latencies), 2),
        "latency_p90_ms": round(sorted(latencies)[int(len(latencies) * 0.9)], 2),
        "latency_p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2),
        "latency_stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
        "per_token_p50_ms": round(statistics.median(token_latencies), 3),
        "per_token_p99_ms": round(sorted(token_latencies)[int(len(token_latencies) * 0.99)], 3),
        "output_tokens": output_len,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    args = parser.parse_args()

    engine = InferenceEngine(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )

    results = benchmark_latency(
        engine=engine,
        num_requests=args.num_requests,
        input_len=args.input_len,
        output_len=args.output_len,
    )

    print("\n" + "=" * 60)
    print("Latency Benchmark Results")
    print("=" * 60)
    for key, val in results.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
