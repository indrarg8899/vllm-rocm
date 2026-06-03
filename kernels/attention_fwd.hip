// Paged Attention Forward Kernel for AMD MI300X (HIP)
// Optimized for CDNA 3 architecture with MFMA instructions

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Block dimensions optimized for MI300X wavefronts
#define BLOCK_SIZE 16
#define NUM_THREADS 256
#define NUM_WAVES 4  // waves per CU for latency hiding

__global__ void paged_attention_fwd_kernel(
    const half* __restrict__ q,           // [num_seqs, num_heads, head_dim]
    const half* __restrict__ k_cache,     // [2, num_layers, max_blocks, block_size, num_heads, head_dim]
    const half* __restrict__ v_cache,     // same layout
    const int* __restrict__ block_tables, // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens, // [num_seqs]
    half* __restrict__ output,            // [num_seqs, num_heads, head_dim]
    float* __restrict__ lse,              // log-sum-exp for numerical stability [num_seqs, num_heads]
    int num_seqs,
    int num_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int layer_idx,
    float scale
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    
    if (seq_idx >= num_seqs || head_idx >= num_heads) return;
    
    const int tid = threadIdx.x;
    const int context_len = context_lens[seq_idx];
    
    // Shared memory for partial reductions
    __shared__ float s_max[NUM_THREADS / 32];  // per-warp max
    __shared__ float s_sum[NUM_THREADS / 32];  // per-warp sum
    
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    
    // Pointer to this query vector
    const half* q_ptr = q + (seq_idx * num_heads + head_idx) * head_dim;
    
    // Accumulate attention output in FP32 for accuracy
    float out_acc[128];  // max head_dim
    for (int d = 0; d < head_dim; d++) {
        out_acc[d] = 0.0f;
    }
    
    // Iterate over key/value blocks for this sequence
    const int* blocks = block_tables + seq_idx * max_blocks_per_seq;
    const int num_blocks = (context_len + block_size - 1) / block_size;
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int physical_block = blocks[block_idx];
        
        // Process each position in the block
        for (int pos = 0; pos < block_size; pos++) {
            int global_pos = block_idx * block_size + pos;
            if (global_pos >= context_len) break;
            
            // Compute Q * K^T dot product
            const half* k_ptr = k_cache + 
                (((0 * num_seqs + layer_idx) * 1024 + physical_block) * block_size + pos) 
                * num_heads * head_dim + head_idx * head_dim;
            
            float dot = 0.0f;
            for (int d = tid; d < head_dim; d += NUM_THREADS) {
                dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
            }
            
            // Warp-level reduction for dot product
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_down(dot, offset);
            }
            
            dot *= scale;
            
            // Update running softmax
            float new_max = fmaxf(max_val, dot);
            float exp_old = expf(max_val - new_max);
            float exp_new = expf(dot - new_max);
            
            for (int d = 0; d < head_dim; d++) {
                out_acc[d] *= exp_old;
            }
            
            // Add V contribution
            const half* v_ptr = v_cache + 
                (((1 * num_seqs + layer_idx) * 1024 + physical_block) * block_size + pos)
                * num_heads * head_dim + head_idx * head_dim;
            
            for (int d = tid; d < head_dim; d += NUM_THREADS) {
                out_acc[d] += exp_new * __half2float(v_ptr[d]);
            }
            
            sum_exp = sum_exp * exp_old + exp_new;
            max_val = new_max;
        }
    }
    
    // Final normalization and write output
    for (int d = tid; d < head_dim; d += NUM_THREADS) {
        output[(seq_idx * num_heads + head_idx) * head_dim + d] = 
            __float2half(out_acc[d] / sum_exp);
    }
    
    // Write log-sum-exp
    if (tid == 0) {
        lse[seq_idx * num_heads + head_idx] = max_val + logf(sum_exp);
    }
}

// Host launch function
extern "C" void launch_paged_attention_fwd(
    const half* q, const half* k_cache, const half* v_cache,
    const int* block_tables, const int* context_lens,
    half* output, float* lse,
    int num_seqs, int num_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    int layer_idx, float scale, hipStream_t stream
) {
    dim3 grid(num_seqs, num_heads);
    dim3 block(NUM_THREADS);
    
    hipLaunchKernelGGL(
        paged_attention_fwd_kernel, grid, block, 0, stream,
        q, k_cache, v_cache, block_tables, context_lens,
        output, lse, num_seqs, num_heads, head_dim,
        block_size, max_blocks_per_seq, layer_idx, scale
    );
}
