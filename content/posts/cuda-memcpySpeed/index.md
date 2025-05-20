---
title: "CUDA Memcpy Speed Validation among three methods"
date: 2024-05-20T21:00:00+09:00
draft: false
tags: ["CUDA", "Memory", "VRAM", "Pinned Memory", "Zero-Copy"]
categories: ["Programming", "GPU", "Memory"]
summary: "In this post, I validate the speed of three memory allocation strategies in CUDA: cudaMalloc, cudaMallocManaged, and  zero-copy memory mapping."
---

let's go