%%writefile 08_Tensor_Cores_Smem_WMMA.cu
// Kernel 8 - Tensor Cores with Shared Memory WMMA
//
// Progression from version 7:
// - Version 7: each warp loads WMMA fragments directly from global memory
// - Version 8: the whole block first stages tiles into shared memory,
//   then each warp loads its own 16x16 fragment from shared memory
//
// Why this matters:
// - better locality than direct global fragment loads
// - less redundant global-memory traffic
// - closer to production Tensor Core kernels
// - practical stepping stone before cp.async / TMA / WGMMA
//
// ─────────────────────────────────────────────────────────────────────────────
// NCU PROFILING — v08 Tensor Cores (Shared Memory WMMA)
// Expected bottleneck: Memory wall shifted slightly - Scalar Global loads
//
// Expected results (M=N=K=4096):
//   sm__warps_active              ~ 40-60%      
//   dram__bytes_read              ~ dramatically LOWER than v07 (L1 cache bypass)
//   Performance                   ~ 10-15 TFLOPS  ← Compute spikes massively
// ─────────────────────────────────────────────────────────────────────────────

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>

using namespace nvcuda;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WM = 32;
constexpr int WN = 64;

constexpr int WARPS_M = BM / WM; // 4
constexpr int WARPS_N = BN / WN; // 2
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N; // 8
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32; // 256

constexpr int FRAGS_M = WM / WMMA_M; // 2
constexpr int FRAGS_N = WN / WMMA_N; // 4

// Padding to reduce SMEM Bank Conflicts during wmma::load_matrix_sync
// 8 __half = 16 bytes = 4 banks offset.
constexpr int PAD = 8; 

__global__ void sgemm_tensor_core_smem_wmma(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Shared memory allocations
    __shared__ __half sA[BM][BK + PAD];
    __shared__ __half sB[BK][BN + PAD];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int warp_row = warp_m * WM;
    const int warp_col = warp_n * WN;

    if (block_row + warp_row >= M || block_col + warp_col >= N) {
        return;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[FRAGS_M][FRAGS_N];

    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {

        // 1. All threads load data from Global into Shared Memory collaboratively
        for (int idx = tid; idx < BM * BK; idx += THREADS_PER_BLOCK) {
            int row = idx / BK;
            int col = idx % BK;
            int g_row = block_row + row;
            int g_col = k0 + col;
            sA[row][col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : __float2half(0.0f);
        }

        for (int idx = tid; idx < BK * BN; idx += THREADS_PER_BLOCK) {
            int row = idx / BN;
            int col = idx % BN;
            int g_row = k0 + row;
            int g_col = block_col + col;
            sB[row][col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : __float2half(0.0f);
        }

        __syncthreads(); // Wait for all loads to complete

        // 2. Consume the shared memory tile (BK = 32 implies 2 steps of WMMA_K=16)
        #pragma unroll
        for (int k_step = 0; k_step < BK; k_step += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[FRAGS_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag[FRAGS_N];

            #pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                int a_row = warp_row + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &sA[a_row][k_step], BK + PAD);
            }

            #pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                int b_col = warp_col + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &sB[k_step][b_col], BN + PAD);
            }

            #pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                #pragma unroll
                for (int j = 0; j < FRAGS_N; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        
        __syncthreads(); // Wait before overwriting SMEM in next cycle
    }

    // 3. Store the result
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            const int c_r = block_row + warp_row + i * WMMA_M;
            const int c_c = block_col + warp_col + j * WMMA_N;
            
            if (c_r < M && c_c < N) {
                #pragma unroll
                for (int t = 0; t < c_frag[i][j].num_elements; t++) {
                    c_frag[i][j].x[t] = c_frag[i][j].x[t] * alpha;
                }

                if (beta != 0.0f) {
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_orig;
                    wmma::load_matrix_sync(c_orig, C + c_r * N + c_c, N, wmma::mem_row_major);
                    #pragma unroll
                    for (int t = 0; t < c_frag[i][j].num_elements; t++) {
                        c_frag[i][j].x[t] = c_frag[i][j].x[t] + beta * c_orig.x[t];
                    }
                }
                wmma::store_matrix_sync(C + c_r * N + c_c, c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

#include "/content/runner_half.h"

void run_08_tensor_core_smem_wmma(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_tensor_core_smem_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        std::cerr << "This WMMA kernel needs Tensor Core capable hardware (SM70+).\n";
        return 1;
    }

    run_benchmark_half(run_08_tensor_core_smem_wmma, M, N, K, "08_Tensor_Core_Smem_WMMA");
    return 0;
}
