---
title: "How to Build libcudf from Source?"
date: 2025-06-04
summary: "This article introduces how to build libcudf, the backend of the RAPIDS library cuDF, from source."
tags: ["CUDA", "GPGPU", "RAPIDS", "libcudf", "CUDF", "AWS", "WSL2", "Ubuntu"]
---

# Table of Contents

- [Introduction](#introduction)
- [What is cuDF?](#what-is-cudf)
- [What is libcudf?](#what-is-libcudf)
- [Motivation](#motivation)
- [Conclusion (First)](#conclusion-first)
- [What I Tried](#what-i-tried)
- [Suspected Causes of Failure](#suspected-causes-of-failure)
- [Conclusion](#conclusion)

---

## Introduction
This article is not a tutorial but a "sharing of failures."
I was trying to build a library called **libcudf**, but ran into several issues when things I expected to work didn't. Through a variety of experiments, I eventually succeeded.

Often, only the final successful process gets shared, while the trial-and-error and failures get discarded. Since I went through many failures this time, I want to document them before I forget.

## What is cuDF?
To explain **libcudf**, we first need to understand **cuDF**.
**cuDF** is one of the packages developed by the **[RAPIDS](https://rapids.ai/)** project.
The **RAPIDS** project primarily uses NVIDIA CUDA and Apache Arrow to accelerate various Python libraries.

For example, the Pandas library, widely used in Python for statistical analysis, has a counterpart in the **RAPIDS** project with the same function names. According to [this report](https://developer.nvidia.com/blog/rapids-cudf-accelerates-pandas-nearly-150x-with-zero-code-changes/), it achieves up to 150x speedup with no code changes.

Although we won't go into detail about **Apache Arrow** here, it greatly simplifies data exchange across different execution environments.

The **RAPIDS** project is structured in two layers: Python and C++.
The backend is written in C++ and CUDA C++, and is provided in compiled form.
The Python wrapper allows the C++ library to be used with the same interface as well-known Python packages.

In this Python package, the component that provides DataFrame functionality is **cuDF**.
The [RAPIDS GitHub repository](https://github.com/rapidsai) is a treasure trove of CUDA implementations for fundamental operations and is highly educational.

## What is libcudf?
**libcudf** is the backend library of cuDF within the RAPIDS project.
It can be cloned from GitHub and built manually, or it can be installed as a bundled part of **cuDF**.

The RAPIDS project provides a well-designed installation method. Using the **[sdkmanager](https://docs.rapids.ai/install/#sdkm:~:text=SDK%20Manager%20(Ubuntu%20Only))**, you can set up the entire environment.

![sdkmanager](sdk_manager.png)

## Motivation
You might ask, “Why build libcudf from source?”
Actually, I have a project that needs to utilize libcudf in C++ for high-speed processing.
If you’re using Python already, simply installing cuDF is fine.

However, if your environment is C++-based or you want to utilize CUDA performance in non-Python environments, converting to binary and referencing it from C++ is often more practical.

Also, relying only on the C++ side helps avoid version compatibility issues. Each of NVIDIA Driver, CUDA Toolkit, Python, C++ build tools, and OS have compatible versions and all must match.
In my experience, managing CUDA versions for Python and its packages is especially troublesome (local setups for image-generation AIs can feel like a labyrinth).

From a performance perspective, skipping Python and directly connecting to the C++ library reduces overhead.
Naturally, this increases development difficulty.

Here’s why I want to use libcudf directly:
1. **The project is based on C++**
2. **The target users are not engineers, so I want to avoid compatibility issues with Python and CUDA Toolkit**
3. **To maximize speed, I want to avoid Python overhead and develop directly in C++**

## Conclusion (First)
With the following setup, the build succeeded in about 20 minutes:
- CPU: Intel Core-i7 14700F
- GPU: NVIDIA RTX 4080
- OS: Windows 11
    - Ubuntu 22.04 on WSL2
- CUDA: 12.6
- cmake: 3.29.6
- gcc: 11.4

Basically, follow the [instructions on cuDF GitHub](https://github.com/rapidsai/cudf/blob/branch-25.08/CONTRIBUTING.md#general-requirements).

## What I Tried
In all the following experiments, the build **FAILED** to proceed:

| Environment | CPU | GPU | OS | CUDA | -jN | Result |
|---|---|---|---|---|---|---|
| Local PC | Ryzen 3700X | NVIDIA RTX 3060 | WSL2 Ubuntu 24.04 | 12.9 | Full | stopped at cicc command for building type_dispatcher.cu.o |
| Local PC | Intel i7-8700 | NVIDIA RTX 4070Ti | WSL2 Ubuntu 24.04 | 12.9 | Full | stopped at cicc command for building type_dispatcher.cu.o |
| Local PC | Intel i7-14700F | NVIDIA RTX 4080 | WSL2 Ubuntu 24.04 | 12.9 | 1 | stopped at cicc command for max.cu.o |
| Local PC | Intel i7-14700F | NVIDIA RTX 4080 | WSL2 Ubuntu 24.04 | 12.9 | Full | stopped at cicc command for building type_dispatcher.cu.o |
| AWS g5.2xlarge | 4C8T | A10G | Ubuntu 24.04 | 12.9 | Full | stopped at cicc command for sum.cu.o |

## Suspected Causes of Failure
The main differences appear to be the Ubuntu version and the CUDA version.
Since the major version of CUDA (12) is the same, I suspect instability within that version range.

In the future, I plan to test by changing everything **except** the CUDA version to confirm if the problem lies elsewhere.
I’ll post a follow-up article if further investigation reveals anything new.

## Conclusion
- libcudf successfully builds on Windows 11 with WSL2
- However, certain combinations of CUDA and Ubuntu versions can cause the build to hang indefinitely. If the build seems stuck for hours, suspect a compatibility issue.

---

Thanks for reading!  
👉 [GitHub Repo (cuda-examples)](https://github.com/yaikeda/cuda-examples/)
