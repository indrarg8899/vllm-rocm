// Rotary Position Embedding (RoPE) Kernel for AMD MI300X (HIP)
// Supports both standard and NTK-aware scaling

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void rotary_embedding_kernel(
    half* __restrict__ q,                 // [num_tokens, num_heads, head_dim]
    half* __restrict__ k,                 // [num_tokens, num_kv_heads, head_dim]
    const float* __restrict__ cos_cache,  // [max_seq_len, head_dim/2]
    const float* __restrict__ sin_cache,  // [max_seq_len, head_dim/2]
    const int* __restrict__ positions,    // [num_tokens]
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float base,
    float scaling_factor
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    
    if (token_idx >= num_tokens || head_idx >= num_heads) return;
    
    const int tid = threadIdx.x;
    const int pos = positions[token_idx];
    const int half_dim = rotary_dim / 2;
    
    // Get cos/sin values for this position
    const float* cos_ptr = cos_cache + pos * half_dim;
    const float* sin_ptr = sin_cache + pos * half_dim;
    
    // Process Q heads
    half* q_ptr = q + (token_idx * num_heads + head_idx) * head_dim;
    
    for (int i = tid; i < half_dim; i += blockDim.x) {
        float cos_val = cos_ptr[i];
        float sin_val = sin_ptr[i];
        
        float x0 = __half2float(q_ptr[i]);
        float x1 = __half2float(q_ptr[i + half_dim]);
        
        // Apply rotation: R(theta) * [x0, x1]
        q_ptr[i] = __float2half(x0 * cos_val - x1 * sin_val);
        q_ptr[i + half_dim] = __float2half(x0 * sin_val + x1 * cos_val);
    }
    
    // Process K heads (if num_kv_heads < num_heads, only heads < num_kv_heads)
    if (head_idx < num_kv_heads) {
        half* k_ptr = k + (token_idx * num_kv_heads + head_idx) * head_dim;
        
        for (int i = tid; i < half_dim; i += blockDim.x) {
            float cos_val = cos_ptr[i];
            float sin_val = sin_ptr[i];
            
            float x0 = __half2float(k_ptr[i]);
            float x1 = __half2float(k_ptr[i + half_dim]);
            
            k_ptr[i] = __float2half(x0 * cos_val - x1 * sin_val);
            k_ptr[i + half_dim] = __float2half(x0 * sin_val + x1 * cos_val);
        }
    }
}

// Precompute cos/sin cache
__global__ void compute_cos_sin_kernel(
    float* __restrict__ cos_cache,  // [max_seq_len, head_dim/2]
    float* __restrict__ sin_cache,
    int max_seq_len,
    int head_dim,
    float base,
    float scaling_factor
) {
    const int pos = blockIdx.x;
    const int i = threadIdx.x;
    
    if (pos >= max_seq_len) return;
    
    int half_dim = head_dim / 2;
    if (i >= half_dim) return;
    
    // NTK-aware scaling: effective base = base * scaling_factor^(2/head_dim)
    float theta = 1.0f / (powf(base, (float)(2 * i) / head_dim));
    theta /= scaling_factor;
    
    float angle = pos * theta;
    cos_cache[pos * half_dim + i] = cosf(angle);
    sin_cache[pos * half_dim + i] = sinf(angle);
}

extern "C" {
    void launch_rotary_embedding(
        half* q, half* k,
        const float* cos_cache, const float* sin_cache,
        const int* positions,
        int num_tokens, int num_heads, int num_kv_heads,
        int head_dim, int rotary_dim, hipStream_t stream
    ) {
        dim3 grid(num_tokens, num_heads);
        dim3 block(min(256, rotary_dim / 2));
        
        hipLaunchKernelGGL(
            rotary_embedding_kernel, grid, block, 0, stream,
            q, k, cos_cache, sin_cache, positions,
            num_tokens, num_heads, num_kv_heads,
            head_dim, rotary_dim, 500000.0f, 1.0f
        );
    }
    
    void launch_compute_cos_sin(
        float* cos_cache, float* sin_cache,
        int max_seq_len, int head_dim,
        float base, float scaling_factor,
        hipStream_t stream
    ) {
        dim3 grid(max_seq_len);
        dim3 block(min(256, head_dim / 2));
        
        hipLaunchKernelGGL(
            compute_cos_sin_kernel, grid, block, 0, stream,
            cos_cache, sin_cache, max_seq_len, head_dim,
            base, scaling_factor
        );
    }
}
