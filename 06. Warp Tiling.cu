%%writefile 06_Warp_Tiling.cu
// kiểu 5 là mỗi thread độc lập 1 block tile lớn. 
// Mỗi block tính tile 128 x 128
// Mỗi thread tính 8 x 8 = 64 output
// Block có 16 x 16 = 256 threads
// sang kiểu 6 là mỗi warp chịu trách nhiệm một sub-tile rõ ràng.
// mỗi block tính tile 64 x 64
// mỗi warp tính 32 x 16
// mỗi thread tính 8 x 2 = 16 output
// block vẫn là 256 threads, nhưng tổ chức thành 8 warps rất rõ ràng

// Result:
// Kernel 6: Warp Tiling
// Block tile: 64x64x8
// Warp tile : 32x16
// Thread tile: 8x2
// Matrix: 512x512x512
// Avg time: 0.259 ms
// Performance: 1035.91 GFLOP/s
// Computing CPU reference...
// Correct: YES (max abs error = 0.00)

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

// Block tile: 128x128x16 (Upscaled to maximize memory-level parallelism)
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;

// Warp tile: 32x64 (since we have 8 warps)
constexpr int WM = 32;
constexpr int WN = 64;

constexpr int WARPS_M = BM / WM;   // 128/32 = 4
constexpr int WARPS_N = BN / WN;   // 128/64 = 2
constexpr int WARP_SIZE = 32;
constexpr int THREADS_PER_BLOCK = WARPS_M * WARPS_N * WARP_SIZE; // 256

// Thread tile: 8x8 (64 acc registers per thread maxes out FMA slots)
constexpr int TM = 8;
constexpr int TN = 8;

__global__ void sgemm_warp_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Padding + 2 to eliminate 100% of write bank conflicts inside the block
    __shared__ float sA[BK][BM + 2]; // Transposed: [k][m]
    __shared__ float sB[BK][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    // Which warp are we in the block?
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    // Mapping the 32 threads inside a warp to the 32x64 Warp Tile
    // lane_row_group = 0..3 (4 rows of 8 threads) -> 4 * 8 (TM) = 32 rows
    // lane_col_group = 0..7 (8 cols of 4 threads) -> 8 * 8 (TN) = 64 cols
    const int lane_row_group = lane / 8; // 0..3
    const int lane_col_group = lane % 8; // 0..7

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int warp_row = warp_m * WM;
    const int warp_col = warp_n * WN;

    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        
        // --- 1. Vectorized Load A ---
        int inner_col_A = tid % (BK / 4);
        int inner_row_A = tid / (BK / 4);
        int stride_A = THREADS_PER_BLOCK / (BK / 4); 

        #pragma unroll
        for (int load_idx = 0; load_idx < (BM * BK) / 1024; ++load_idx) {
            int row_a = inner_row_A + load_idx * stride_A;
            int col_a = inner_col_A * 4;
            
            // Bounds checking (simplified for exact multiples of 128)
            int g_row = block_row + row_a;
            int g_col = k0 + col_a;
            
            float4 val_a;
            if (g_row < M && g_col < K) {
                val_a = reinterpret_cast<const float4*>(&A[g_row * K + g_col])[0];
            } else {
                val_a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            
            sA[col_a + 0][row_a] = val_a.x;
            sA[col_a + 1][row_a] = val_a.y;
            sA[col_a + 2][row_a] = val_a.z;
            sA[col_a + 3][row_a] = val_a.w;
        }

        // --- 2. Vectorized Load B ---
        int inner_col_B = tid % (BN / 4);
        int inner_row_B = tid / (BN / 4);
        int stride_B = THREADS_PER_BLOCK / (BN / 4);

        #pragma unroll
        for (int load_idx = 0; load_idx < (BK * BN) / 1024; ++load_idx) {
            int row_b = inner_row_B + load_idx * stride_B;
            int col_b = inner_col_B * 4;
            
            int g_row = k0 + row_b;
            int g_col = block_col + col_b;
            
            float4 val_b;
            if (g_row < K && g_col < N) {
                val_b = reinterpret_cast<const float4*>(&B[g_row * N + g_col])[0];
            } else {
                val_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            
            reinterpret_cast<float4*>(&sB[row_b][col_b])[0] = val_b;
        }

        __syncthreads();

        // --- 3. Compute Phase ---
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float regA[TM];
            float regB[TN];

            // Load sA (Transposed scalar broadcasts to eliminate conflicts)
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int row = warp_row + lane_row_group * TM + i;
                regA[i] = sA[kk][row];
            }

            // Load sB (float4 vectorized reads)
            float4 b0 = reinterpret_cast<float4*>(&sB[kk][warp_col + lane_col_group * TN])[0];
            float4 b1 = reinterpret_cast<float4*>(&sB[kk][warp_col + lane_col_group * TN + 4])[0];
            
            regB[0] = b0.x; regB[1] = b0.y; regB[2] = b0.z; regB[3] = b0.w;
            regB[4] = b1.x; regB[5] = b1.y; regB[6] = b1.z; regB[7] = b1.w;

            // FMA math
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // --- 4. Store Phase to C ---
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = block_row + warp_row + lane_row_group * TM + i;
        if (row >= M) continue;

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = block_col + warp_col + lane_col_group * TN + j;
            if (col < N) {
                C[row * N + col] = alpha * acc[i][j] + beta * C[row * N + col];
            }
        }
    }
}

#include "performance_runner/runner.h"

void run_06_warp_tiled(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_warp_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    run_benchmark(run_06_warp_tiled, M, N, K, "06_Warp_Tiling_4096");
    return 0;
}
