# Multi-GPU Setup

## Prerequisites

- Multiple AMD GPUs (MI300X or MI250X)
- RCCL installed
- GPUs connected via NVLink/IF Fabric

## Configuration

```bash
# Set RCCL environment
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export HSA_OVERRIDE_GFX_VERSION=9.0a  # MI300X

# Launch with tensor parallelism
python -m vllm_rocm.serve \
    --model meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1
```

## Recommended TP Sizes

| GPU Count | Model | Precision |
|---|---|---|
| 1 | Llama-3-8B | FP16 |
| 2 | Llama-3-70B | FP8 |
| 4 | Llama-3-70B | FP16 |
| 8 | Mixtral-8x7B | FP16 |

## Troubleshooting

- If RCCL hangs, set `NCCL_P2P_DISABLE=1`
- For multi-node, set `MASTER_ADDR` and `MASTER_PORT`
