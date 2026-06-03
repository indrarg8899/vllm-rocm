# Architecture

## System Overview

vLLM-ROCm implements a high-throughput serving system for large language models on AMD Instinct MI300X GPUs, inspired by the [vLLM](https://github.com/vllm-project/vllm) project.

## Key Components

### 1. API Server (`src/api_server.py`)

FastAPI-based HTTP server implementing the OpenAI-compatible chat completions API:

```
POST /v1/chat/completions   — Chat-style generation
POST /v1/completions        — Raw text completion
GET  /v1/models             — List loaded models
GET  /health                — Health check
```

### 2. Inference Engine (`src/engine.py`)

Core orchestrator coordinating all subsystems:

- Loads models from HuggingFace Hub via `ModelLoader`
- Initializes PagedAttention KV-cache
- Manages the continuous-batching scheduler
- Handles request lifecycle (tokenization → generation → detokenization)

### 3. Scheduler (`src/scheduler.py`)

Implements **continuous batching** — sequences are added to batches dynamically rather than waiting for an entire batch to finish.

**Policies:**
- **FCFS** (First Come First Served) — default
- **Priority** — priority-weighted scheduling

**Preemption:**
- **Recompute** — free KV-cache, re-generate when scheduled again
- **Swap** — move KV-cache to CPU, restore when GPU frees up

### 4. PagedAttention (`src/cache/paged_attention.py`)

Inspired by virtual memory. Key-value tensors are stored in fixed-size **blocks** rather than contiguous buffers.

Benefits:
- No memory fragmentation
- Dynamic allocation as sequences grow
- Prefix sharing between sequences
- Efficient GPU memory utilization

### 5. Block Manager (`src/cache/block_manager.py`)

Physical memory management:
- **Free list** — tracks available GPU memory blocks
- **Reference counting** — enables prefix sharing
- **Swap** — CPU ↔ GPU block movement for preemption

## Memory Layout

```
GPU Memory
┌──────────────────────────────────┐
│ Model Weights (static)           │
├──────────────────────────────────┤
│ KV Cache Blocks (paged)          │
│  ┌─────┐ ┌─────┐ ┌─────┐       │
│  │K₀,V₀│ │K₁,V₁│ │K₂,V₂│ ...  │
│  │      │ │      │ │      │      │
│  └─────┘ └─────┘ └─────┘       │
├──────────────────────────────────┤
│ Activation Buffers               │
├──────────────────────────────────┤
│ Temporary (NCCL, etc.)           │
└──────────────────────────────────┘
```

## Data Flow

```
Client Request
    │
    ▼
┌───────────────┐
│  API Server   │ Parse request, queue to scheduler
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Scheduler   │ Assign blocks, manage batching
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  KV Cache     │ Read/Write PagedAttention blocks
│  Manager      │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  Model        │ Forward pass, produce next token
│  (HIP/CK)     │
└───────┬───────┘
        │
        ▼
    Repeat until EOS / max_tokens
```

## ROCm-Specific Optimizations

| Optimization | Description |
|---|---|
| HIP Graphs | CUDA-graph equivalent for ROCm — captures kernel launch sequences |
| CK Flash Attention | Composable Kernel flash-attention backend |
| Triton Kernels | Custom ROCm-optimized Triton kernels for attention |
| FP8 E4M3 | Native MI300X FP8 quantization for 2× throughput |
| XNACK | Unified memory via ROCm page migration |
