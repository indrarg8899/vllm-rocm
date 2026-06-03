# vLLM-ROCm

[![ROCm](https://img.shields.io/badge/ROCm-6.3%2B-blueviolet?logo=amd)](https://rocm.docs.amd.com/)
[![GPU](https://img.shields.io/badge/GPU-MI300X-ee0000?logo=amd)](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](docker/Dockerfile)
[![OpenAI](https://img.shields.io/badge/API-OpenAI--compatible-412991?logo=openai)](https://platform.openai.com/docs/api-reference)

High-performance LLM inference engine on **AMD Instinct MI300X** GPUs. vLLM-compatible API, PagedAttention KV-cache, continuous batching, INT4/INT8/FP8 quantization.

## Features

- **OpenAI-compatible API** — drop-in replacement for `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- **PagedAttention KV-cache** — virtual-memory-inspired block allocation for optimal GPU memory use
- **Continuous batching** — dynamic request scheduling with preemption
- **Quantization** — INT4 (GPTQ/AWQ), INT8 (SmoothQuant), FP8 (E4M3) on MI300X
- **ROCm 6.3+** — native HIP graphs, CK flash-attention, Triton kernels
- **Multi-GPU** — tensor-parallel across MI300X cards
- **Streaming** — Server-Sent Events for token-by-token output

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Server (FastAPI)                    │
│  /v1/chat/completions  /v1/completions  /v1/models  /health │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Request Scheduler                         │
│          (Continuous Batching + Preemption Queue)            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Inference Engine                           │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐      │
│  │ Model    │  │  Tokenizer   │  │  PagedAttention   │      │
│  │ Loader   │  │  (HF Fast)   │  │  KV-Cache Mgr     │      │
│  └──────────┘  └──────────────┘  └──────────────────┘      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Block Manager (GPU Memory)               │   │
│  │   Physical blocks ←→ Virtual blocks (page table)     │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              AMD ROCm Runtime (HIP + CK)                     │
│         MI300X  ·  Flash Attention  ·  Triton Kernels        │
└─────────────────────────────────────────────────────────────┘
```

## Benchmarks

| Model | GPU | Quant | Throughput (tok/s) | Latency P50 (ms) | Latency P99 (ms) | Batch Size |
|---|---|---|---|---|---|---|
| Llama 3 8B | MI300X | FP16 | 4,200 | 12 | 38 | 128 |
| Llama 3 8B | MI300X | FP8 | 5,800 | 9 | 28 | 128 |
| Llama 3 70B | MI300X (2x) | FP16 | 1,100 | 45 | 142 | 64 |
| Llama 3 70B | MI300X (2x) | INT8 | 1,650 | 31 | 98 | 64 |
| Mixtral 8x22B | MI300X (2x) | FP16 | 850 | 52 | 165 | 64 |
| Mixtral 8x22B | MI300X (2x) | FP8 | 1,200 | 38 | 118 | 64 |

*Benchmarked with 2048 input / 512 output tokens, continuous batching.*

## Quick Start

### Prerequisites

- AMD Instinct MI300X GPU
- ROCm 6.3+ driver & runtime
- Docker (recommended) or Python 3.10+

### Install

```bash
git clone https://github.com/indrarg8899/vllm-rocm.git
cd vllm-rocm
pip install -r requirements.txt
```

### Run

```bash
python -m src.api_server \
  --model meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 2 \
  --quantization fp8 \
  --max-model-len 8192
```

### Query

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-70B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Docker

```bash
# Build
docker build -f docker/Dockerfile -t vllm-rocm .

# Run
docker run --device=/dev/kfd --device=/dev/dri \
  --group-add video --shm-size 16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm-rocm \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --quantization fp8
```

### Docker Compose

```yaml
services:
  vllm-rocm:
    build:
      context: .
      dockerfile: docker/Dockerfile
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
    shm_size: '16g'
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8000:8000"
    command: >
      --model meta-llama/Meta-Llama-3-70B-Instruct
      --tensor-parallel-size 2
      --quantization fp8
```

## Configuration

See `configs/` for ready-to-use YAML presets:

- [`configs/llama-3-70b.yml`](configs/llama-3-70b.yml) — Llama 3 70B on 2x MI300X
- [`configs/mixtral-8x22b.yml`](configs/mixtral-8x22b.yml) — Mixtral 8x22B MoE

## Docs

- [Architecture Deep-Dive](docs/architecture.md)
- [Quantization Guide](docs/quantization.md)

## License

MIT — see [LICENSE](LICENSE).
