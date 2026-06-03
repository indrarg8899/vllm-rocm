"""Memory manager for GPU KV cache and model weights."""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MemoryPool:
    name: str
    total_bytes: int
    allocated_bytes: int = 0
    device: str = "cuda:0"

    @property
    def utilization(self) -> float:
        return self.allocated_bytes / self.total_bytes if self.total_bytes > 0 else 0.0

    @property
    def free_bytes(self) -> int:
        return self.total_bytes - self.allocated_bytes

    def format_size(self, bytes_val: int) -> str:
        gb = bytes_val / (1024 ** 3)
        if gb >= 1:
            return f"{gb:.1f} GB"
        mb = bytes_val / (1024 ** 2)
        return f"{mb:.1f} MB"


class GPUResourceManager:
    """Manages GPU memory for KV cache and model weight allocation."""

    def __init__(
        self,
        device: str = "cuda:0",
        max_memory_fraction: float = 0.92,
    ):
        self.device = device
        self.max_memory_fraction = max_memory_fraction
        self.pools: Dict[str, MemoryPool] = {}
        self._allocated_tensors: Dict[str, torch.Tensor] = {}

    def get_gpu_memory_info(self) -> Dict[str, int]:
        """Query GPU memory via ROCm/HIP."""
        if not torch.cuda.is_available():
            return {"total": 0, "free": 0, "used": 0}

        total = torch.cuda.get_device_properties(self.device).total_mem
        free = torch.cuda.mem_get_info(self.device)[0]
        return {
            "total": total,
            "free": free,
            "used": total - free,
        }

    def create_memory_pool(
        self,
        name: str,
        num_bytes: int,
    ) -> MemoryPool:
        pool = MemoryPool(
            name=name,
            total_bytes=num_bytes,
            device=self.device,
        )
        self.pools[name] = pool
        return pool

    def allocate_tensor(
        self,
        name: str,
        shape: tuple,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        self._allocated_tensors[name] = tensor
        return tensor

    def estimate_kv_cache_size(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
    ) -> int:
        bytes_per_element = torch.finfo(dtype).bits // 8
        kv_per_block = (
            2  # key + value
            * num_heads
            * block_size
            * head_dim
            * bytes_per_element
        )
        return kv_per_block * num_blocks

    def get_optimal_config(
        self,
        model_size_gb: float,
        dtype: torch.dtype = torch.float16,
        max_batch_tokens: int = 8192,
    ) -> Dict[str, int]:
        info = self.get_gpu_memory_info()
        total = info["total"]
        usable = int(total * self.max_memory_fraction)

        bytes_per_element = torch.finfo(dtype).bits // 8
        model_bytes = int(model_size_gb * (1024 ** 3))
        remaining = max(usable - model_bytes, 0)

        block_size = 16
        kv_per_block = 2 * 32 * block_size * 128 * bytes_per_element
        num_blocks = remaining // kv_per_block

        return {
            "total_memory": total,
            "model_memory": model_bytes,
            "kv_cache_memory": num_blocks * kv_per_block,
            "num_blocks": num_blocks,
            "block_size": block_size,
            "usable_memory": usable,
        }

    def get_status(self) -> Dict:
        info = self.get_gpu_memory_info()
        return {
            "gpu_memory": info,
            "pools": {
                name: {
                    "utilization": f"{p.utilization:.1%}",
                    "allocated": p.format_size(p.allocated_bytes),
                    "total": p.format_size(p.total_bytes),
                }
                for name, p in self.pools.items()
            },
            "allocated_tensors": len(self._allocated_tensors),
        }
