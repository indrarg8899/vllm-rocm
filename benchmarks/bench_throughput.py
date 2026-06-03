"""Throughput benchmark for vLLM-ROCm inference engine."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class BenchResult:
    num_requests: int
    throughput_tok_s: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    total_time_s: float
    input_tokens: int
    output_tokens: int


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> tuple[float, dict]:
    """Send a single completion request and return latency + response."""
    start = time.perf_counter()
    resp = await client.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        },
        timeout=300.0,
    )
    latency = time.perf_counter() - start
    data = resp.json()
    return latency, data


async def run_bench(
    url: str,
    model: str,
    num_requests: int,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
) -> BenchResult:
    """Run concurrent benchmark."""
    # Generate input prompt
    prompt = " ".join(["hello world"] * (input_tokens // 2))

    latencies: list[float] = []
    total_tokens = 0

    sem = asyncio.Semaphore(concurrency)

    async def bounded_request():
        async with sem:
            return await send_request(client, url, model, prompt, output_tokens)

    async with httpx.AsyncClient() as client:
        start_time = time.perf_counter()
        tasks = [bounded_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_time

    valid_results = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
            continue
        lat, data = r
        latencies.append(lat)
        try:
            total_tokens += data["usage"]["completion_tokens"]
        except (KeyError, TypeError):
            total_tokens += 0
        valid_results.append((lat, data))

    if not latencies:
        return BenchResult(num_requests=0, throughput_tok_s=0, latency_p50_ms=0,
                          latency_p90_ms=0, latency_p99_ms=0, total_time_s=0,
                          input_tokens=input_tokens, output_tokens=output_tokens)

    latencies.sort()
    return BenchResult(
        num_requests=len(valid_results),
        throughput_tok_s=total_tokens / total_time if total_time > 0 else 0,
        latency_p50_ms=latencies[len(latencies) // 2] * 1000,
        latency_p90_ms=latencies[int(len(latencies) * 0.9)] * 1000,
        latency_p99_ms=latencies[int(len(latencies) * 0.99)] * 1000,
        total_time_s=total_time,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def main():
    parser = argparse.ArgumentParser(description="vLLM-ROCm Throughput Benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()

    print(f"=== vLLM-ROCm Benchmark ===")
    print(f"URL:         {args.url}")
    print(f"Model:       {args.model}")
    print(f"Requests:    {args.num_requests}")
    print(f"Input:       {args.input_tokens} tokens")
    print(f"Output:      {args.output_tokens} tokens")
    print(f"Concurrency: {args.concurrency}")
    print()

    result = asyncio.run(run_bench(args.url, args.model, args.num_requests,
                                   args.input_tokens, args.output_tokens, args.concurrency))

    print(f"--- Results ---")
    print(f"Completed:     {result.num_requests}/{args.num_requests}")
    print(f"Throughput:    {result.throughput_tok_s:.1f} tok/s")
    print(f"Latency P50:   {result.latency_p50_ms:.1f} ms")
    print(f"Latency P90:   {result.latency_p90_ms:.1f} ms")
    print(f"Latency P99:   {result.latency_p99_ms:.1f} ms")
    print(f"Total time:    {result.total_time_s:.2f} s")

    # Save JSON
    out = {
        "num_requests": result.num_requests,
        "throughput_tok_s": round(result.throughput_tok_s, 1),
        "latency_p50_ms": round(result.latency_p50_ms, 1),
        "latency_p90_ms": round(result.latency_p90_ms, 1),
        "latency_p99_ms": round(result.latency_p99_ms, 1),
        "total_time_s": round(result.total_time_s, 2),
        "config": {"input_tokens": args.input_tokens, "output_tokens": args.output_tokens, "concurrency": args.concurrency},
    }
    with open("bench_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to bench_result.json")


if __name__ == "__main__":
    main()
