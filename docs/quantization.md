# Quantization Guide

vLLM-ROCm supports three quantization formats optimized for AMD Instinct MI300X GPUs.

## Supported Formats

| Format | Bits | Method | MI300X Support | Quality Impact |
|---|---|---|---|---|
| FP8 | 8 | E4M3 native | ✅ Native | Minimal |
| INT8 | 8 | SmoothQuant | ✅ Via CK | Low |
| INT4 | 4 | Group-wise | ✅ Via CK | Moderate |
| AWQ | 4 | Activation-aware | ✅ | Low-Moderate |
| GPTQ | 4 | Layer-wise | ✅ | Low-Moderate |

## FP8 Quantization

FP8 E4M3 is the recommended format for MI300X — it runs at 2× throughput vs FP16 on MI300X matrix cores.

```bash
python -m src.api_server \
  --model meta-llama/Meta-Llama-3-70B-Instruct \
  --quantization fp8 \
  --tensor-parallel-size 2
```

**How it works:**
1. Weights are cast to FP8 E4M3 with per-tensor scaling
2. FP8 matrix multiplications use MI300X's native FP8 matrix cores
3. Accumulation happens in FP32 for numerical stability
4. Dequantization restores FP16 outputs for attention and residual connections

**Best for:** Production inference where throughput matters more than peak quality.

## INT8 (SmoothQuant)

SmoothQuant redistributes the quantization difficulty between activations and weights.

```bash
python -m src.api_server \
  --model meta-llama/Meta-Llama-3-70B-Instruct \
  --quantization int8
```

**How it works:**
1. Per-channel smoothing factor α is computed from activation statistics
2. Smoothed weights are quantized per-channel to INT8
3. Inference uses INT8 GEMM via CK (Composable Kernels)
4. Output is dequantized back to FP16

**Best for:** Models where FP8 causes quality degradation.

## INT4 (AWQ / GPTQ)

### AWQ (Activation-Aware Weight Quantization)

```bash
python -m src.api_server \
  --model TheBloke/Llama-2-70B-chat-AWQ \
  --quantization awq \
  --tensor-parallel-size 2
```

### GPTQ

```bash
python -m src.api_server \
  --model TheBloke/Llama-2-70B-chat-GPTQ \
  --quantization gptq \
  --tensor-parallel-size 2
```

**How it works:**
1. Groups of 128 weights share a scaling factor
2. Each weight is quantized to 4 bits (-8 to 7 range)
3. Dequantization: `weight_fp16 = weight_int4 * scale`
4. Group-wise quantization minimizes precision loss

**Best for:** Memory-constrained deployments (saves ~75% vs FP16).

## Pre-quantized Models

You can load pre-quantized models from HuggingFace directly:

```bash
# Pre-quantized AWQ
python -m src.api_server --model TheBloke/Mixtral-8x22B-v0.1-AWQ --quantization awq

# Pre-quantized GPTQ
python -m src.api_server --model TheBloke/Llama-3-70B-GPTQ --quantization gptq
```

## Benchmark Comparison

| Model | FP16 | FP8 | INT8 | INT4 |
|---|---|---|---|---|
| Llama 3 8B (tok/s) | 4,200 | 5,800 | 5,100 | 7,200 |
| Llama 3 70B (tok/s) | 1,100 | 1,650 | 1,400 | 2,000 |
| Llama 3 8B Memory (GB) | 16 | 9 | 9 | 5.5 |
| Llama 3 70B Memory (GB) | 140 | 75 | 75 | 42 |

*Throughput measured at batch_size=128 on MI300X, input=2048, output=512.*

## Recommendations

| Scenario | Recommended Format |
|---|---|
| Maximum throughput | FP8 |
| Best quality/throughput tradeoff | FP8 |
| Maximum context length | INT4 (AWQ) |
| Smallest memory footprint | INT4 (GPTQ) |
| Multi-GPU inference | FP8 or INT8 |
