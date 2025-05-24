---
title: "CUDA Streaming and Overlap Visualization with Nsight"
date: 2025-05-23T21:00:00+09:00
draft: false
tags: ["CUDA", "Stream", "Nsight", "OpenCV", "Parallel"]
categories: ["Programming", "GPU"]
summary: "This post explains how to implement asynchronous parallel processing using CUDA streams and how to visualize GPU execution overlap with Nsight Systems."
---

## Table of Contents

* Purpose
* Why Asynchronous Processing?
* Implementation Flow
  * Introducing `CudaImageResource`
  * Key Points of Asynchronization
  * Processing Structure
  * Memory Allocation Differences
* Visualization with Nsight
* Key Learnings

## Purpose

In the previous implementation, memory transfers (HtoD and DtoH) and kernel execution were performed synchronously. That means the data transfer to GPU, kernel execution, and memory copy back were strictly ordered and waited on each other.

While GPU itself performs internal parallelism, operations like memory transfer are still executed synchronously.

This time, we use `cudaStream_t` to issue instructions asynchronously to the GPU.

## Why Asynchronous Processing?

When processing multiple images, the HtoD transfer, kernel execution, and DtoH transfer for each image must occur in sequence. However, these steps for different images do not need to be synchronized.

For example, while the kernel is executing for image 1, you can already transfer image 2 to the GPU. Similarly, while DtoH transfer is happening for image 1, the kernel for image 2 can be running.

If this pipeline works ideally, it means the kernel executions finish while memory transfers are happening, reducing total processing time.

Although this kind of pipelining is not unique to CUDA, achieving it correctly is challenging. In this post, we experiment with **parallel scheduling using CUDA Streams**, a fundamental technique in CUDA.

## Implementation Flow

We improve the previous sequential CUDA implementation by enabling **asynchronous parallel processing using `cudaStream_t`**. We also use **Nsight Systems** to check whether operations actually run in parallel on the GPU.

### 1. Introducing `CudaImageResource`

Since we're working with asynchronous processing, resource management becomes more important.

In synchronous processing, you could release memory after processing one image and then allocate resources for the next. But in async, you must be sure that GPU operations are done before releasing any memory.

To ensure this, we use a struct to manage `cudaArray*`, `cudaTextureObject_t`, `unsigned char*`, and `cudaStream_t` per image. We then call `Destroy()` after all work is complete.

```cpp
struct CudaImageResource {
    cudaStream_t stream = nullptr;
    cudaArray* cuArray = nullptr;
    cudaTextureObject_t texObj = 0;
    unsigned char* d_output = nullptr;
    cv::Mat output;

    void Destroy() {
        if (texObj) cudaDestroyTextureObject(texObj);
        if (cuArray) cudaFreeArray(cuArray);
        if (d_output) cudaFree(d_output);
        if (stream) cudaStreamDestroy(stream);
    }
};
```

### 2. Key Points of Asynchronization

We used the following asynchronous functions:

* `cudaMemcpy2DToArrayAsync()`
* `grayscaleKernel<<<..., ..., 0, stream>>>`
* `cudaMemcpyAsync()`

These allow us to queue tasks in each stream without waiting for previous steps.

### 3. Processing Structure

Here's how we structured the processing:

* For all images: allocate memory and create texture objects (not async)
* Then: do transfer, kernel, and result copy **asynchronously per stream**
* Wait for all streams using `cudaStreamSynchronize()`
* Save results and clean up with `Destroy()`

By doing this, we ensure only async-safe operations go into the streams, which maximizes overlap.

### 4. Memory Allocation Differences

In our testing, we found that when allocating memory using `cudaArray`, streams did **not** run in parallel. All processing occurred sequentially.

The root cause is unclear, but it may be due to the internal handling of `cudaArray`, which differs from standard device memory.

When we switched to using regular `cudaMalloc` and copied the image data into memory ourselves, stream overlap worked as expected.

## Visualization with Nsight

Here are the results visualized with Nsight Systems:

![nsight sequential](nsight_sequential.png)

<div style="text-align: center">
  <i>Sequential Processing</i><br>
</div>

In the above result, although streams are defined separately, all processing happens sequentially — no true overlap.

![nsight streaming](nsight_streaming.png)

<div style="text-align: center">
  <i>Asynchronous Processing</i><br>
</div>

In this result, **memory transfers and kernel executions overlap clearly**. The green and dark blue bars partially overlap.

Although we didn't heavily optimize the kernel or memory pipeline, this partial overlap already shows the benefit. With further tuning, overlap can be improved significantly.

---

## Key Learnings

* One stream per image with dedicated resources is a practical design pattern
* `cudaMalloc` and `cudaCreateTextureObject` are not asynchronous; prepare them beforehand
* Nsight Systems helps visualize asynchronous behavior intuitively
* Resource management is easier and safer when encapsulated in a struct or class

---

Thanks for reading!
👉 [GitHub Repo (cuda-examples)](https://github.com/yaikeda/cuda-examples/)
