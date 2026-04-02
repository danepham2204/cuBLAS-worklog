// %%cuda
// ─────────────────────────────────────────────────────────────────────────────
// NCU PROFILING — v07 Tensor Cores (Global Memory Baseline)
// Expected bottleneck: Memory Wall - VRAM Bandwidth Starvation
// Why it stalls: Without Shared Memory, each Warp redundantly hits Global Memory
//                competing directly for DRAM, destroying the 65 TFLOPs ideal limit.
//
// Expected results (M=N=K=4096):
//   sm__warps_active              ~ 95%+        ← Warps are completely stalled waiting for VRAM
//   dram__bytes_read              ~ high GB     ← Global memory redundant loads are massive
//   Performance                   ~ 2-3 TFLOPS  ← Compute starved
// ─────────────────────────────────────────────────────────────────────────────

// Kernel 07 - Tensor Cores (Global Baseline)
// This version introduces wmma::mma_sync hardware Tensor Cores.
// It skips Shared Memory to demonstrate the Memory Wall of pulling fragments 
// straight from VRAM, but structures the blocks and warps perfectly (128x128x16)
// so each warp maximizes its mathematical throughput (32x64 Warp Tile).

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>

using namespace nvcuda;

constexpr int BM = 128;
constexpr int BN = 128;

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

__global__ void sgemm_tensor_core_wmma(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    
    // Which warp are we in the block
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int warp_row = block_row + warp_m * WM;
    const int warp_col = block_col + warp_n * WN;

    if (warp_row >= M || warp_col >= N) {
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

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[FRAGS_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag[FRAGS_N];

        // Load 2 fragments of A straight from Global Memory
        #pragma unroll
        for (int i = 0; i < FRAGS_M; ++i) {
            const int a_row = warp_row + i * WMMA_M;
            wmma::load_matrix_sync(a_frag[i], A + a_row * K + k0, K);
        }

        // Load 4 fragments of B straight from Global Memory
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            const int b_col = warp_col + j * WMMA_N;
            wmma::load_matrix_sync(b_frag[j], B + k0 * N + b_col, N);
        }

        // Issue 8 MMA Operations
        #pragma unroll
        for (int i = 0; i < FRAGS_M; ++i) {
            #pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }
    }

    // Write Back C
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            const int c_r = warp_row + i * WMMA_M;
            const int c_c = warp_col + j * WMMA_N;
            
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

#include "performance_runner/runner_half.h"

void run_07_tensor_core_wmma(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_tensor_core_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        std::cerr << "This WMMA kernel needs Tensor Core capable hardware (SM70+).\n";
        return 1;
    }

    run_benchmark_half(run_07_tensor_core_wmma, M, N, K, "07_Tensor_Core_WMMA_4096");
    return 0;
}
