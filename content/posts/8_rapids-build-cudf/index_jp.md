---
title: "libcudfをソースからビルドする方法"
date: 2025-06-04
summary: "RAPIDSライブラリのcuDFのバックエンドであるlibcudfをソースからビルドする方法を紹介します。"
tags: ["CUDA", "GPGPU", "RAPIDS", "libcudf", "CUDF", "AWS", "WSL2", "Ubuntu"]
---

# 目次

- [はじめに](#はじめに)
- [cuDFとは何か](#cudfとは何か)
- [libcudfとは何か](#libcudfとは何か)
- [モチベーション](#モチベーション)
- [（先に）結論](#先に結論)
- [試したこと](#試したこと)
- [うまくいかなかった原因の予想](#うまくいかなかった原因の予想)
- [結論](#結論)

---

## はじめに
今回の記事は、チュートリアルではなく「失敗の共有」です。
**libcudf** というライブラリをビルドしていたのですが、できると思っていたことができず、様々な方向から実験をして最終的に成功しました。
最終的にうまくいった内容というのは往々にしてうまくいったことだけが共有され、失敗した内容というのは捨てられがちです。
今回は、せっかくいろいろな失敗をしたので、この記憶がなくなる前に情報をまとめておきます。

## cuDFとは何か
**libcudf** の説明をするためには、まず **cuDF** から説明しなければなりません。
**cuDF** とは、**[RAPIDS](https://rapids.ai/)** プロジェクトで開発されたパッケージの一つです。
**RAPIDS** プロジェクトは、NVIDIA CUDAとApache Arrowを基本的に使用し、Pythonの様々なライブラリを高速化するプロジェクトです。
例えば、Pythonで統計解析に多用されるPandasライブラリは、同じ関数名で**RAPIDS**プロジェクトに実装されており、実に[150倍の高速化を達成](https://developer.nvidia.com/blog/rapids-cudf-accelerates-pandas-nearly-150x-with-zero-code-changes/)したという報告もあります。

**Appache Arrow** についてはこの記事では大きく取り上げませんが、使用することで異なる実行環境間でのデータ受け渡しがスムーズになります。

この**RAPIDS**プロジェクトは大きくPythonとC++の二つの階層に分かれます。
バックエンドはC++とCUDA C++で書かれ、コンパイルされた状態で提供されます。
このC++ライブラリを既存の著名なPythonパッケージと同じ方式で呼び出せるようにしたラッパー部分がPythonで記述されます。

このPythonパッケージにおいて、DataFrameの機能を提供するのが、**cuDF** です。
[RAPIDSプロジェクトのGitHub](https://github.com/rapidsai)は基礎的な処理を行うCUDA実装の宝庫でもあり、非常に参考になります。

## libcudfとは何か
**libcudf**は、RAPIDSプロジェクト内のcuDFのバックエンド側のライブラリの名称です。
GitHubページからCloneして自分でビルドすることもできますし、**cuDF**をインストールしたときにバンドルしてインストールすることもできます。

RAPIDSプロジェクトはインストール方法も非常に整っており、**[sdkmanager](https://docs.rapids.ai/install/#sdkm:~:text=SDK%20Manager%20(Ubuntu%20Only))** というものを使用すると、環境自体は用意できます。

![sdkmanager](sdk_manager.png)

## モチベーション
私は、以下の点でlibcudfを直接使いたいと思っています。

```
1. C++を前提とした開発プロジェクトである
2. 想定するユーザがエンジニアではないため、PythonとCUDA Toolkit周辺の相性問題をスキップしたい
3. 可能な限り高速に処理したいので、Pythonのオーバーヘッドを考えず開発したい
```

ここで一つ疑問があるでしょう。「私はなぜlibcudfをソースからビルドするのか？」
実は、私はlibcudfをC++から使って高速化したいプロジェクトがあります。

すでにPythonを使っている人が処理を高速化するために使う分には普通にインストールすればOKです。
しかし、C++が前提の環境や、C++やPython以外の環境にCUDAの処理速度を役立てるためには、一旦バイナリ形式に変換した後で参照するほうが使い勝手が良いです。

また、C++側のライブラリのみに依存することで、バージョンの相性問題を1段階回避できます。
NVIDIA Driver、CUDA Toolkit、Python、C++ビルド、OSはそれぞれ対応するバージョンがあり、すべての条件を満たす必要があります。
特に経験上、Pythonと、Python内の各パッケージが使用するCUDA Toolkitバージョンを正しく設定するのは手間がかかります。（画像生成系AIのローカル環境開発は迷宮のようです）

速度の観点でも、Pythonを一段挟むよりも、C++ライブラリに直接接続するほうがオーバーヘッドが削減できます。
この場合、開発の難易度は当然増加します。


## （先に）結論
以下の構成であれば、20分程度でビルドに成功した。
- CPU: Intel Core-i7 14700F
- GPU: NVIDIA RTX 4080
- OS: Windows 11
    - Ubuntu 22.04 on WSL2
- CUDA: 12.6
- cmake: 3.29.6
- gcc: 11.4

基本的には、[cudfのGitHubの説明](https://github.com/rapidsai/cudf/blob/branch-25.08/CONTRIBUTING.md#general-requirements)に合わせればOK。

## 試したこと
以下の実験ではすべてビルドが進まなくなりました。

| Environment | CPU | GPU | OS | CUDA | -jN | Result |
|---|---|---|---|---|---|---|
| Local PC | Ryzen 3700X | NVIDIA RTX 3060 | WSL2 Ubuntu 24.04 | 12.9 | Full | stopped at cicc command for buildingtype_dispatcher.cu.o |
| Local PC | Intel i7-8700 | NVIDIA RTX 4070Ti | WSL2 Ubuntu 24.04 | 12.9 | Full | stopped at cicc command for buildingtype_dispatcher.cu.o |
| Local PC | Intel i7-14700F | NVIDIA RTX 4080 | WSL2 Ubuntu 24.04 | 12.9 | 1 | stopped at cicc command for max.cu.o |
| Local PC | Intel i7-14700F | NVIDIA RTX 4080 | WSL2 Ubuntu 24.04 | 12.9 | Full | stopped at cicc command for buildingtype_dispatcher.cu.o |
| AWS g5.2xlarge | 4C8T | A10G | Ubuntu 24.04 | 12.9 | Full | stopped at cicc command for sum.cu.o |

## うまくいかなかった原因の予想
特に変わった点はUbuntuのバージョンと、CUDAのバージョンです。特にCUDAのメジャーバージョン（12）は変わっていないことから、CUDAのバージョンが安定していないのではないかと考えています。
また後日、CUDAバージョン以外を最新の環境に合わせて実験してビルドできるかを確認すれば、何が問題であったか明らかになるでしょう。
今後さらに調査を進めたらまた記事を投稿しようと思います。

## 結論
- libcudfは WIndows 11のWSL2上でもビルドに成功した
- ただし、CUDAやUbuntuのバージョンによって無限に終わらなくなるパターンがあるため、数時間ビルドが停止していたら相性問題を疑ったほうが良い。

---

## Reference
- [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
