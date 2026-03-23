# How to Run on Google Colab

## Environment
- Platform: Google Colab (T4 GPU, SM75)
- CUDA: 12.8
- Compiler: nvcc

---

## Part 1 — Run kernel benchmark

All steps are Colab notebook cells. Each file is written to disk using `%%writefile` — no file uploads needed.

---

### Step 1: Check CUDA compiler (run once per session)

```python
!nvcc --version
```

Verifies nvcc is available. Colab T4 has it pre-installed.

---

### Step 2: Write runner.h to disk

```python
%%writefile runner.h
#pragma once
// ... contents of performance_runner/runner.h ...
```

The kernel files contain `#include "/content/runner.h"`. This step writes the header to `/content/runner.h` so nvcc can find it at compile time. For kernels 07–10 (FP16), write `runner_half.h` the same way.

---

### Step 3: Write the kernel to disk

```python
%%writefile 06_Warp_Tiling.cu
// ... contents of 06. Warp Tiling.cu ...
```

Jupyter cannot compile CUDA directly from a cell. `%%writefile` saves the cell content as a real `.cu` file on disk for nvcc to read.

---

### Step 4: Compile

```python
!nvcc -O3 -arch=sm_75 -lcublas 06_Warp_Tiling.cu -o 06_warp
```

| Flag | Purpose |
|---|---|
| `-O3` | Maximum compiler optimization |
| `-arch=sm_75` | **Required** for T4. Without it, nvcc defaults to SM52 (Maxwell) — WMMA intrinsics compile away silently and produce garbage results |
| `-lcublas` | Links cuBLAS so `runner.h` can call `cublasSgemm` as the reference |
| `-o 06_warp` | Output binary name |

---

### Step 5: Run

```python
!./06_warp
```

`runner.h` automatically: warms up once → runs 20 iterations → computes avg time → prints GFLOP/s → validates correctness against cuBLAS.

**Sample output:**
```
06_Warp_Tiling   | Size: 2048x2048x2048 | Time: 13.326 ms | Perf: 1289.19 GFLOP/s | Max Err: 1.83e-04
```

---

## Part 2 — Profile with ncu (find bottleneck)

`ncu` and `runner.h` are independent. There is no need to run the benchmark first. `ncu` wraps directly around the compiled binary.

---

### Step 6: Unlock profiling permissions (run once per session)

Colab blocks GPU hardware performance counters by default. The fix depends on the driver version of your runtime:

**Option A — Older Colab runtimes:**
```python
!sudo sh -c 'echo 1 > /proc/driver/nvidia/params/RmProfilingAdminOnly'
```

**Option B — Newer Colab runtimes (path above does not exist):**
```python
# No file to write — pass bypass flags directly to ncu instead (see Step 7)
```

If you see `sh: cannot create /proc/driver/nvidia/params/RmProfilingAdminOnly: Directory nonexistent`, skip this step and use the `--target-processes all --clock-control none` flags in Step 7 instead.

---

### Step 7: Profile

```python
!ncu --target-processes all --clock-control none --kernel-name sgemm_warp_tiled --launch-skip 1 --launch-count 1 --metrics sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum ./06_warp
```

**Important:**
- All metrics must be on a single line with no spaces after commas. Colab line continuations (`\`) break `ncu` argument parsing — it interprets the next line as the binary name instead of a metric.
- `--kernel-name` filters to your kernel only (skips the cuBLAS reference call from runner.h). **If you profile other versions (e.g. 07), make sure to change this to the corresponding kernel name (like `sgemm_tensor_core_wmma`).**
- `--launch-skip 1` skips the warmup invocation, `--launch-count 1` profiles only one run instead of all 20. Without these, ncu profiles every invocation and takes 20x longer.

`--target-processes all` và `--clock-control none` bypass the permission requirement on newer Colab runtimes where the `/proc` path does not exist.

**Fallback — use `nvprof` if ncu still fails:**
```python
!nvprof --metrics sm_efficiency,dram_read_throughput,flop_count_sp ./06_warp
```
`nvprof` is the legacy profiler and requires no special permissions. Metric names differ slightly from ncu but cover the same hardware events.

| Metric | Measures | Reveals |
|---|---|---|
| `sm__warps_active` | % of warps actually running | Thread idle time, undersized tiles |
| `l1tex__t_bytes...global_op_ld` | Total bytes read from global memory | Memory pressure, cache miss rate |
| `sm__sass...ffma_pred_on` | Actual FMA instruction count | Compute underutilization |
| `dram__bytes_read` | DRAM traffic | Memory bandwidth saturation |

> **Note:** `ncu` injects instrumentation and slows the kernel 10–100x. Never use the execution time reported by `ncu` to compare performance — use only the metrics.

---

### Step 8 (optional): Export to CSV for multi-kernel comparison

```python
!ncu --csv --log-file profile_06.csv --target-processes all --clock-control none --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum ./06_warp
```

Useful when comparing v05 vs v06 vs v09 side by side.

---

## Summary

```
%%writefile runner.h          →  write benchmark harness to /content/runner.h
%%writefile kernel.cu         →  write kernel source to disk
nvcc -O3 -arch=sm_75 ...      →  compile to binary
./binary                      →  runner.h: wall-clock GFLOP/s + correctness check
ncu --metrics ... ./binary    →  hardware counters: why fast or slow
```

## Common mistakes

| Mistake | Effect |
|---|---|
| Missing `-arch=sm_75` | WMMA intrinsics silently compile away; kernel runs empty and reports fake throughput |
| Skipping `%%writefile runner.h` | Compile error — `#include "/content/runner.h"` not found |
| `/proc/driver/nvidia/...` path not found | Newer Colab driver — skip Step 6 and add `--target-processes all --clock-control none` to ncu instead |
| `ERR_NVGPUCTRPERM` after bypass flags | Fall back to `nvprof` which needs no permission |
| Using ncu execution time for perf comparison | Wrong — instrumentation overhead makes it 10–100x slower than real runtime |
| Using `runner.h` for kernels 07–10 | Type mismatch — kernels 07–10 take `__half*` inputs; use `runner_half.h` instead |
