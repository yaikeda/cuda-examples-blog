---
title: "Comparing CPU and GPU Performance in Vector Addition with CUDA"
date: 2024-05-18T21:00:00+09:00
draft: false
tags: ["CUDA", "GPU", "Benchmark", "C++", "nvcc"]
categories: ["Programming", "GPU", "Optimization"]
summary: "A first benchmark comparing CPU and GPU performance in vector addition using CUDA. Includes code samples, timing analysis, and lessons learned."
---

## Introduction

In this post, I explore the performance characteristics of a simple vector addition task implemented on both CPU and GPU using CUDA.  
This experiment is part of my hands-on learning in GPU programming and performance profiling.

---

## Environment

- **CUDA Toolkit**: 12.6.20
- **Platform**: Windows 11
- **GPU**: NVIDIA RTX 3060  
- **Profiler**: `cudaEventElapsedTime`, `std::chrono`

---

## Implementation

### CPU Version

```cpp
void vectorAddCPU(const float* A, const float* B, float* C, const int N) {
    for (int i = 0; i < N; ++i)
        C[i] = A[i] + B[i];
}
```

### GPU Version
```cpp
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
```

## Benchmark Results

I tested both versions across varying data sizes.
★ is faster 

| N (Elements)     | CPU Time (ms) | GPU Time (ms) |
|------------------|---------------|----------------|
| 512              | ★0.0009        | 0.5263         |
| 1,000,000        | 1.5718        | ★0.5168         |
| 1,000,000,000    | 2670.95       | ★36.4035        |


---

## Analysis

- For small arrays (≤ 512), the CPU performs faster due to GPU kernel launch overhead. Memory transfer overhead is ignored in this evaluation.
- Around N = 1 million, GPU starts to outperform CPU due to its massive parallelism.
- Kernel execution is extremely fast on GPU, but data transfer cost can be significant for smaller problems.

---

## Learned

- Always guard against out-of-bounds access in CUDA kernels using `if (i < N)`.
- For real-time or low-latency systems, overlapping transfers (`cudaMemcpyAsync`) with kernel execution may be essential.
- CUDA features
    - *\_\_global\_\_* : Indicates the device code
    - *cudaMalloc* : GPU malloc()
    - *cudaMemcpy* : Copy data between host and device
    - *cudaMemcpyAAAToBBB* : Defines the mode of transfer direction
    - *cudaEventCreate* : Add mark to check some timing
    - *cudaEventSynchronize* : Wait for the end of device computation
    - *cudaEventElapsedTime* : Get elapsed time with GPU time
    - *cudaFree* : GPU free()

---

## GitHub

👉 [CUDA Examples GitHub Repository](https://github.com/yaikeda/cuda-examples)

---

Thanks for reading!
