---
title: "Follow-Up: Building libcudf on WSL2"
date: 2025-06-17
summary: "Continuing from the previous post, this article explains how I replicated successful libcudf builds by cloning WSL2 environments and installing the correct CUDA Toolkit."
tags: ["CUDA", "libcudf", "WSL2", "Ubuntu", "CUDA Toolkit", "NVIDIA"]
---

## Background
This post continues from [the earlier entry](../8_rapids-build-cudf/) about building **libcudf**. In that article I reported that the build succeeded without fully understanding why. After further investigation, I managed to reproduce the success in a more systematic way. Here I share that process.

## Prerequisites
I limited my testing to the Windows Subsystem for Linux (WSL2), which had worked at the end of the previous experiment. The host is Windows with an NVIDIA RTX GPU, running Ubuntu 24.04 inside WSL2.

## Method
I experimented with multiple setups inside WSL2 without using Docker. WSL2 offers `import` and `export` commands that allow you to duplicate an entire distribution.

### Cloning a WSL2 Distribution
Install your distribution with:

```bash
wsl --install <Distro>
```

where `<Distro>` is something like `Ubuntu-24.04`. Once the initial installation completes, export the minimal state so you can restore it later:

```bash
wsl --export <Distro> <FileName>
```

To create a copy from the exported file, run:

```bash
wsl --import <Distro> <InstallLocation> <FileName>
```

You can choose any location, including another drive, when importing. Using this feature I compared several Ubuntu environments.

## Conclusion
The issue turned out to be installing `cuda-toolkit` from Ubuntu's official repository, which only provides the outdated version 12.0. By switching to NVIDIA's repository, I could install newer versions such as 12.6 or 12.8. At the time of writing, 12.6 is considered stable, so it is a safe choice unless you need a different version.

## Resources
The shell scripts I used for building libcudf are available in the [cudf-build-scripts](https://github.com/yaikeda/cudf-build-scripts) repository. Feel free to adapt them for your own setup.

## Insights
This experiment showed the limits of relying solely on Ubuntu's official repositories. Through trial and error I learned more about how CUDA versions interact with drivers. I also came to appreciate CMake much more—it resolves CUDA-related linking smoothly and is far easier than writing `gcc` or `nvcc` commands by hand. The scripts mentioned above include how to install any required CMake version.

