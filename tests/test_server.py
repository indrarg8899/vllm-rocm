"""Tests for vLLM-ROCm API server."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx


@pytest.fixture
def mock_engine():
    engine = AsyncMock()
    engine.model_name = "test-model"
    engine.gpu_memory_usage.return_value = 42.5
    engine.generate_chat.return_value = "Hello! How can I help?"
    engine.generate_text.return_value = "The capital of France is Paris."
    return engine


class TestConfig:
    def test_default_config(self):
        from src.config import EngineConfig
        cfg = EngineConfig()
        assert cfg.model == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert cfg.tensor_parallel_size == 1
        assert cfg.quantization is None
        assert cfg.block_size == 16

    def test_custom_config(self):
        from src.config import EngineConfig
        cfg = EngineConfig(model="custom/model", tensor_parallel_size=4, quantization="fp8")
        assert cfg.model == "custom/model"
        assert cfg.tensor_parallel_size == 4
        assert cfg.quantization == "fp8"


class TestBlockManager:
    def test_alloc_free(self):
        from src.config import EngineConfig
        from src.cache.block_manager import BlockManager
        cfg = EngineConfig(num_kv_cache_blocks=100, block_size=16)
        bm = BlockManager(config=cfg)
        initial_free = bm.free_blocks

        blocks = bm.allocate(10, seq_id="test")
        assert len(blocks) == 10
        assert bm.free_blocks == initial_free - 10

        bm.free(blocks)
        assert bm.free_blocks == initial_free

    def test_cannot_over_allocate(self):
        from src.config import EngineConfig
        from src.cache.block_manager import BlockManager
        cfg = EngineConfig(num_kv_cache_blocks=5, block_size=16)
        bm = BlockManager(config=cfg)
        with pytest.raises(ValueError, match="Cannot allocate"):
            bm.allocate(10)


class TestPagedAttention:
    def test_write_read_kv(self):
        import torch
        from src.config import EngineConfig
        from src.cache.block_manager import BlockManager
        from src.cache.paged_attention import PagedAttentionKVCacheManager

        cfg = EngineConfig(num_kv_cache_blocks=10, block_size=16)
        bm = BlockManager(config=cfg)
        kv = PagedAttentionKVCacheManager(config=cfg, block_manager=bm)
        kv.init_cache(num_layers=2, num_heads=8, head_dim=64, device="cpu")

        blocks = bm.allocate(2)
        # Write at position 0, layer 0
        key = torch.randn(8, 1, 64)
        val = torch.randn(8, 1, 64)
        kv.write_kv(blocks, position=0, layer_idx=0, key=key, value=val)

        # Read back
        keys, vals = kv.read_kv(blocks, seq_len=1, layer_idx=0)
        assert keys.shape == (8, 1, 64)
        assert vals.shape == (8, 1, 64)
        assert torch.allclose(keys[0, 0, :], key[0, 0, :], atol=1e-5)


class TestScheduler:
    def test_scheduler_creation(self):
        from src.config import EngineConfig
        from src.scheduler import Scheduler, SeqStatus
        from unittest.mock import MagicMock

        cfg = EngineConfig()
        mock_kv = MagicMock()
        mock_bm = MagicMock()
        mock_bm.free_blocks = 1000
        sched = Scheduler(config=cfg, kv_cache=mock_kv, block_manager=mock_bm)
        assert sched.num_waiting == 0
        assert sched.num_running == 0


class TestQuantizer:
    def test_int8_quantize(self):
        import torch
        from src.models.quantizer import _quantize_int8
        layer = torch.nn.Linear(64, 64, bias=False)
        quantized = _quantize_int8(layer)
        assert quantized.weight.dtype == torch.int8


class TestAPIEndpoints:
    def test_health(self):
        from fastapi.testclient import TestClient
        from src.api_server import app, engine as _eng

        # The lifespan doesn't run in TestClient without the engine
        # Test with mocked lifespan
        client = TestClient(app, raise_server_exceptions=False)
        # Without engine started, health should still return or error gracefully
        resp = client.get("/health")
        assert resp.status_code in (200, 503)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
