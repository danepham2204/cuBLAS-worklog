%%writefile 05_Vectorized_Register_Tiling.cu
// run with nvcc4jupyter extension
#include "/content/runner.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

const int BM = 128;
const int BN = 128;
const int BK = 16;
const int TM = 8;
const int TN = 8;

__global__ void sgemm_vectorized_kernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sA[BK][BM];    // Transposed: [k][m] — stride 128 is 16-byte aligned for float4
    __shared__ float sB[BK][BN];

    float threadResults[TM][TN] = {};
    float regA[TM];
    float regB[TN];

    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int k = 0; k < K; k += BK) {
        int inner_col_A = tid % (BK / 4);
        int inner_row_A = tid / (BK / 4);
        int stride_A = 256 / (BK / 4);

        for (int load_idx = 0; load_idx < (BM * BK) / 1024; ++load_idx) {
            int row_a = inner_row_A + load_idx * stride_A;
            int col_a = inner_col_A * 4;
            float4 val_a = reinterpret_cast<const float4*>(&A[(cRow + row_a) * K + (k + col_a)])[0];
            sA[col_a + 0][row_a] = val_a.x;
            sA[col_a + 1][row_a] = val_a.y;
            sA[col_a + 2][row_a] = val_a.z;
            sA[col_a + 3][row_a] = val_a.w;
        }

        int inner_col_B = tid % (BN / 4);
        int inner_row_B = tid / (BN / 4);
        int stride_B = 256 / (BN / 4);

        for (int load_idx = 0; load_idx < (BK * BN) / 1024; ++load_idx) {
            int row_b = inner_row_B + load_idx * stride_B;
            int col_b = inner_col_B * 4;
            float4 val_b = reinterpret_cast<const float4*>(&B[(k + row_b) * N + (cCol + col_b)])[0];
            reinterpret_cast<float4*>(&sB[row_b][col_b])[0] = val_b;
        }

        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load from sA to Registers (Transposed) - SCALAR READ is mandatory for Broadcast
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                regA[i] = sA[dotIdx][threadIdx.y * TM + i];

            float4 b0 = reinterpret_cast<float4*>(&sB[dotIdx][threadIdx.x * TN])[0];
            float4 b1 = reinterpret_cast<float4*>(&sB[dotIdx][threadIdx.x * TN + 4])[0];
            regB[0] = b0.x; regB[1] = b0.y; regB[2] = b0.z; regB[3] = b0.w;
            regB[4] = b1.x; regB[5] = b1.y; regB[6] = b1.z; regB[7] = b1.w;

            for (int rm = 0; rm < TM; ++rm) {
                for (int rn = 0; rn < TN; ++rn) {
                    threadResults[rm][rn] += regA[rm] * regB[rn];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; j += 4) {
            float4 res;
            res.x = threadResults[i][j + 0];
            res.y = threadResults[i][j + 1];
            res.z = threadResults[i][j + 2];
            res.w = threadResults[i][j + 3];
            int out_r = cRow + threadIdx.y * TM + i;
            int out_c = cCol + threadIdx.x * TN + j;
            reinterpret_cast<float4*>(&C[out_r * N + out_c])[0] = res;
        }
    }
}

#include "/content/runner.h"

void run_05_vectorized(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim(N / BN, M / BM);
    sgemm_vectorized_kernel<<<gridDim, blockDim>>>((float*)d_A, (float*)d_B, d_C, M, N, K);
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    run_benchmark(run_05_vectorized, M, N, K, "05_Vectorized_Register_Tiling_4096");
    return 0;
}
