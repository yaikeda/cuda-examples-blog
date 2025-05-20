---
title: "CUDA Memory Transfer Timing: malloc, managed, and zero-copy"
date: 2025-05-20T21:00:00+09:00
draft: false
tags: ["CUDA", "Memory", "Transfer", "Benchmark"]
categories: ["Programming", "GPU", "Performance"]
summary: "I measured the transfer time of 2GB memory between CPU and GPU using cudaMalloc, cudaMallocManaged, and cudaHostAllocMapped (Zero-Copy)."
---

## Overview

In this session, I explored how memory transfer times vary across several CUDA memory allocation strategies by measuring the time it takes to transfer **2GB** of data using:

- `cudaMalloc` (device memory)
- `cudaMallocManaged` (unified memory)
- `cudaHostAllocMapped` (zero-copy)

All timings were measured by a custom **RAII-style C++ timer class** that automatically logs elapsed time when leaving scope.

## Measurement Setup

Each experiment allocates 2GB of memory and transfers it either:

- from host to device,
- from device to host,
- or allows device to directly access host memory (zero-copy).

The elapsed time is measured using `std::chrono::high_resolution_clock`.

---

## Results

```
cudaMalloc HtoD Elapsed time: 186.46 ms
cudaMalloc DtoH Elapsed time: 213.094 ms
cudaMallocManaged HtoD PrefetchAsync Elapsed time: 0.0199 ms
cudaMallocManaged HtoD Call kernel Elapsed time: 1655.28 ms
Refer unified_ptr from CPU: o
cudaMallocManaged DtoH Elapsed time: 0.6585 ms
Zero-Copy HtoD Elapsed time: 82.9358 ms
host_ptr[0] = 0
Zero-Copy DtoH Elapsed time: 0.0522 ms
```

---

## Observations

- ✅ **`cudaMalloc`** provides stable and reasonable transfer speeds in both directions.  
- ⚠️ **`cudaMallocManaged`** performs very poorly when a GPU kernel attempts to access the data. This is likely due to page fault–based migration via unified virtual memory. The kernel's actual execution time is currently not isolated.
- ⚠️ **Prefetching** unified memory with `cudaMemPrefetchAsync()` is extremely fast—but this may be due to optimization bypassing the actual transfer (e.g., no page migration because the memory is already resident).
- ⚠️ **Zero-Copy** memory appears very fast, especially for device-to-host (DtoH) reads. However, it is unclear whether some hardware optimization or caching is hiding the true latency.

---

## Technical Notes

- Memory size: **2GB** (`1ULL << 31`)
- Custom class `AutoTimeLogger` logs time upon destruction, simplifying measurement.
- Experiments were built and run using `nvcc` and MSVC on Windows 11 with a GeForce RTX 3060.

---

## Next Steps

To improve the precision and fairness of future measurements:

- 🔍 Use **`cudaEventRecord()`** to measure *only* GPU-side kernel execution time, separating transfer from computation.
- ⚖️ Use the **same kernel** across all memory types to enable fair comparison.
- 🧠 Design **access patterns that defeat GPU-side caching** to better reflect raw transfer performance.

---

Thanks for reading!  
👉 [GitHub Repo (cuda-examples)](https://github.com/yaikeda/cuda-examples/)
