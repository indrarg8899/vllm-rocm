// Fused SiLU Activation Kernel for AMD MI300X (HIP)
// Fuses elementwise operations to reduce memory bandwidth usage

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// SiLU (Swish) activation: x * sigmoid(x) = x / (1 + exp(-x))
__global__ void fused_silu_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    // Vectorized load for better memory throughput
    for (int i = idx; i < n / 4; i += stride) {
        const half2* in2 = reinterpret_cast<const half2*>(input + i * 4);
        half2 x0 = in2[0];
        half2 x1 = in2[1];
        
        half2 out0, out1;
        
        // SiLU on each element
        out0.x = x0.x * __float2half(1.0f / (1.0f + expf(-__half2float(x0.x))));
        out0.y = x0.y * __float2half(1.0f / (1.0f + expf(-__half2float(x0.y))));
        out1.x = x1.x * __float2half(1.0f / (1.0f + expf(-__half2float(x1.x))));
        out1.y = x1.y * __float2half(1.0f / (1.0f + expf(-__half2float(x1.y))));
        
        half2* out2 = reinterpret_cast<half2*>(output + i * 4);
        out2[0] = out0;
        out2[1] = out1;
    }
    
    // Handle remainder
    for (int i = (n / 4) * 4 + idx; i < n; i += stride) {
        float x = __half2float(input[i]);
        output[i] = __float2half(x / (1.0f + expf(-x)));
    }
}

// Fused SiLU + gate multiply (for gated FFN): silu(gate) * up
__global__ void fused_silu_gate_kernel(
    const half* __restrict__ gate,     // gating projection
    const half* __restrict__ up,       // up projection
    half* __restrict__ output,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < n / 2; i += stride) {
        const half2* gate2 = reinterpret_cast<const half2*>(gate + i * 2);
        const half2* up2 = reinterpret_cast<const half2*>(up + i * 2);
        
        half2 g = gate2[0];
        half2 u = up2[0];
        
        half2 result;
        result.x = __float2half(
            (__half2float(g.x) / (1.0f + expf(-__half2float(g.x)))) * __half2float(u.x)
        );
        result.y = __float2half(
            (__half2float(g.y) / (1.0f + expf(-__half2float(g.y)))) * __half2float(u.y)
        );
        
        half2* out2 = reinterpret_cast<half2*>(output + i * 2);
        out2[0] = result;
    }
}

extern "C" {
    void launch_fused_silu(
        const half* input, half* output, int n,
        hipStream_t stream
    ) {
        int threads = 256;
        int blocks = min((n + threads * 4 - 1) / (threads * 4), 65535);
        hipLaunchKernelGGL(fused_silu_kernel, dim3(blocks), dim3(threads), 0, stream,
                           input, output, n);
    }
    
    void launch_fused_silu_gate(
        const half* gate, const half* up, half* output, int n,
        hipStream_t stream
    ) {
        int threads = 256;
        int blocks = min((n + threads * 2 - 1) / (threads * 2), 65535);
        hipLaunchKernelGGL(fused_silu_gate_kernel, dim3(blocks), dim3(threads), 0, stream,
                           gate, up, output, n);
    }
}
