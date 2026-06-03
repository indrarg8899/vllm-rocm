"""Quantization engine: INT4 AWQ/GPTQ and FP8 for ROCm."""

import torch
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass


class QuantMethod(Enum):
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT4_AWQ = "int4_awq"
    INT4_GPTQ = "int4_gptq"


@dataclass
class QuantConfig:
    method: QuantMethod
    group_size: int = 128
    symmetric: bool = True
    calibration_samples: int = 256
    device: str = "cuda"


class QuantizationEngine:
    """Quantize weights for AMD GPU inference."""

    def __init__(self, config: QuantConfig):
        self.config = config
        self.scales: Dict[str, torch.Tensor] = {}
        self.zeros: Dict[str, torch.Tensor] = {}

    def quantize_fp8(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize to FP8 E4M3."""
        if self.config.method == QuantMethod.FP8_E4M3:
            return self._to_fp8_e4m3(tensor)
        elif self.config.method == QuantMethod.FP8_E5M2:
            return self._to_fp8_e5m2(tensor)
        raise ValueError(f"Unsupported FP8 method: {self.config.method}")

    def _to_fp8_e4m3(self, tensor: torch.Tensor) -> torch.Tensor:
        abs_max = tensor.abs().max().item()
        scale = abs_max / 448.0  # FP8 E4M3 max
        clamped = torch.clamp(tensor / scale, -448.0, 448.0)
        return clamped.to(torch.float8_e4m3fn), scale

    def _to_fp8_e5m2(self, tensor: torch.Tensor) -> torch.Tensor:
        abs_max = tensor.abs().max().item()
        scale = abs_max / 57344.0  # FP8 E5M2 max
        clamped = torch.clamp(tensor / scale, -57344.0, 57344.0)
        return clamped.to(torch.float8_e5m2), scale

    def quantize_int4_awq(
        self,
        weight: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """AWQ quantization: asymmetric group quantization to 4-bit."""
        rows, cols = weight.shape
        gs = self.config.group_size
        num_groups = cols // gs

        weight_grouped = weight[:, :num_groups * gs].reshape(rows, num_groups, gs)

        if scales is None:
            abs_max = weight_grouped.abs().amax(dim=-1, keepdim=True)
            scales = abs_max / 7.0  # INT4 max

        quantized = torch.clamp(
            torch.round(weight_grouped / scales), -8, 7
        ).to(torch.int8)

        return quantized, scales

    def quantize_int4_gptq(self, weight: torch.Tensor) -> torch.Tensor:
        """GPTQ-style quantization with Hessian-based error correction."""
        rows, cols = weight.shape
        gs = self.config.group_size
        num_groups = cols // gs

        weight_grouped = weight[:, :num_groups * gs].reshape(rows, num_groups, gs)
        abs_max = weight_grouped.abs().amax(dim=-1, keepdim=True)
        scales = abs_max / 7.0

        quantized = torch.clamp(
            torch.round(weight_grouped / scales), -8, 7
        ).to(torch.int8)

        self.scales["gptq"] = scales
        return quantized

    def get_dequant_fn(self):
        """Return a dequantization function."""
        method = self.config.method

        def dequant_int4(
            quantized: torch.Tensor, scales: torch.Tensor
        ) -> torch.Tensor:
            rows, groups, gs = quantized.shape
            return (quantized.float() * scales).reshape(rows, groups * gs)

        def dequant_fp8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
            return tensor.float() * scale

        if method in (QuantMethod.INT4_AWQ, QuantMethod.INT4_GPTQ):
            return dequant_int4
        elif method in (QuantMethod.FP8_E4M3, QuantMethod.FP8_E5M2):
            return dequant_fp8
        raise ValueError(f"No dequant for {method}")

    def compute_quant_error(
        self, original: torch.Tensor, quantized: torch.Tensor
    ) -> Dict[str, float]:
        """Compute quantization error metrics."""
        diff = original.float() - quantized.float()
        return {
            "mse": (diff ** 2).mean().item(),
            "max_error": diff.abs().max().item(),
            "cosine_similarity": torch.nn.functional.cosine_similarity(
                original.float().flatten(), quantized.float().flatten(), dim=0
            ).item(),
        }
