"""Dataclass-based configuration with CLI argument parsing."""
from __future__ import annotations

import argparse
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EngineConfig:
    """Core engine configuration."""

    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    quantization: Optional[str] = None  # fp8 | int8 | int4 | awq | gptq
    dtype: str = "auto"  # auto | float16 | bfloat16 | float32
    block_size: int = 16
    num_kv_cache_blocks: int = 0  # 0 = auto
    swap_space_gb: float = 4.0
    enforce_eager: bool = False
    trust_remote_code: bool = False
    seed: int = 0
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    preemption_mode: str = "recompute"  # recompute | swap
    scheduler_policy: str = "fcfs"  # fcfs | priority
    model_name: str = ""
    huggingface_hub_token: Optional[str] = None

    @classmethod
    def from_cli(cls, args: list[str] | None = None) -> "EngineConfig":
        parser = argparse.ArgumentParser(description="vLLM-ROCm Engine")
        parser.add_argument("--model", type=str, default=cls.model)
        parser.add_argument("--tensor-parallel-size", type=int, default=cls.tensor_parallel_size)
        parser.add_argument("--pipeline-parallel-size", type=int, default=cls.pipeline_parallel_size)
        parser.add_argument("--max-model-len", type=int, default=cls.max_model_len)
        parser.add_argument("--gpu-memory-utilization", type=float, default=cls.gpu_memory_utilization)
        parser.add_argument("--quantization", type=str, default=cls.quantization, choices=[None, "fp8", "int8", "int4", "awq", "gptq"])
        parser.add_argument("--dtype", type=str, default=cls.dtype)
        parser.add_argument("--block-size", type=int, default=cls.block_size)
        parser.add_argument("--num-kv-cache-blocks", type=int, default=cls.num_kv_cache_blocks)
        parser.add_argument("--swap-space-gb", type=float, default=cls.swap_space_gb)
        parser.add_argument("--enforce-eager", action="store_true")
        parser.add_argument("--trust-remote-code", action="store_true")
        parser.add_argument("--seed", type=int, default=cls.seed)
        parser.add_argument("--max-num-batched-tokens", type=int, default=cls.max_num_batched_tokens)
        parser.add_argument("--max-num-seqs", type=int, default=cls.max_num_seqs)
        parser.add_argument("--preemption-mode", type=str, default=cls.preemption_mode)
        parser.add_argument("--scheduler-policy", type=str, default=cls.scheduler_policy)
        parser.add_argument("--hf-token", type=str, default=None)
        parser.add_argument("--config", type=str, default=None, help="YAML config file")
        parsed = parser.parse_args(args)

        # Override from YAML if provided
        if parsed.config:
            with open(parsed.config) as f:
                yml = yaml.safe_load(f)
            for k, v in (yml or {}).items():
                attr = k.replace("-", "_")
                if hasattr(parsed, attr) and getattr(parsed, attr) is None:
                    setattr(parsed, attr, v)

        model_name = parsed.model.split("/")[-1]
        return cls(
            model=parsed.model,
            tensor_parallel_size=parsed.tensor_parallel_size,
            pipeline_parallel_size=parsed.pipeline_parallel_size,
            max_model_len=parsed.max_model_len,
            gpu_memory_utilization=parsed.gpu_memory_utilization,
            quantization=parsed.quantization,
            dtype=parsed.dtype,
            block_size=parsed.block_size,
            num_kv_cache_blocks=parsed.num_kv_cache_blocks,
            swap_space_gb=parsed.swap_space_gb,
            enforce_eager=parsed.enforce_eager,
            trust_remote_code=parsed.trust_remote_code,
            seed=parsed.seed,
            max_num_batched_tokens=parsed.max_num_batched_tokens,
            max_num_seqs=parsed.max_num_seqs,
            preemption_mode=parsed.preemption_mode,
            scheduler_policy=parsed.scheduler_policy,
            model_name=model_name,
            huggingface_hub_token=parsed.hf_token,
        )


@dataclass
class ServerConfig:
    """HTTP server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    @classmethod
    def from_cli(cls) -> "ServerConfig":
        parser = argparse.ArgumentParser(description="vLLM-ROCm Server")
        parser.add_argument("--host", type=str, default=cls.host)
        parser.add_argument("--port", type=int, default=cls.port)
        parser.add_argument("--log-level", type=str, default=cls.log_level)
        parsed, _ = parser.parse_known_args()
        return cls(host=parsed.host, port=parsed.port, log_level=parsed.log_level)
