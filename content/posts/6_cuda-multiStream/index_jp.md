---
title: "CUDAストリーミング処理とNsightによるオーバーラップ可視化"
date: 2025-05-23T21:00:00+09:00
draft: false
tags: ["CUDA", "Stream", "Nsight", "OpenCV", "Parallel"]
categories: ["Programming", "GPU"]
summary: "CUDAのストリーム機能を用いた非同期並列処理と、それをNsight Systemsで可視化する方法について解説します。"
---

## Table of Contents
* 目的
* 非同期処理は何がうれしいのか？
* 実装の流れ
  * `CudaImageResource` の導入
  * 非同期化のポイント
  * 処理の構成
* Nsightでの可視化結果
* 学んだこと

## 目的

前回までのプログラムでは、HtoDのメモリ転送、カーネルの実行、DtoHへのメモリの書き戻しを同期的に行っていました。
同期的とは、コンピュータのデータ転送が終わってからカーネル処理をする、カーネル処理が終わってから計算結果をGPUからメモリに書き戻すという順番を守って処理をしているということです。
この場合でも、GPUでは内部で並列処理をしていますが、転送処理等は同期的に実行されています。
今回は、cudaStreamを使用することで、GPUへ投入した命令を非同期に実行します。

## 非同期処理は何がうれしいのか？

複数の画像を処理する場合、メモリをHtoD転送する処理、カーネルの実行、メモリをDtoH転送する処理は、1枚の画像については順番に実行する必要がありますが、画像間で同期的に実行する必要はありません。
つまり、１枚目の画像を送った後、カーネルの実行中に２枚目の画像を転送、１枚目の画像のDtoH転送している間に２枚目のカーネル実行をしても問題ありません。
このような処理が理想的に動作した場合は、すべての転送が順番に動作している間にカーネル処理が終了しており、全体の処理時間が短縮できます。
CUDAに限らずこういったパイプライン処理を理想的に実装するのは簡単ではありません。
今回はCUDAにおける要素技術であるStreamを使って**複数の処理を並列にスケジューリングする**実験をしてみます。

## 実装の流れ
今回は、これまでの逐次的なCUDA処理を改善し、**CUDAのストリーム（`cudaStream_t`）を用いた非同期並列処理**を実装します。
また、**Nsight Systemsを用いて、実際にGPU上で処理が並列に動作しているかを可視化**する方法も紹介します。

### 1. `CudaImageResource` の導入

CUDA Streamを使う際には非同期処理が必要となるため、リソースの管理も気を付けて行う必要があります。
同期的な処理をする場合には、すべての処理が順番に行われるため、１枚目の画像を処理したらリソースを破棄し、２枚目の画像のリソースを確保して処理をする、という対応ができました。
しかし非同期処理になると、GPUに実行させる命令を投入した後、処理が終了したことを保証したうえでリソースを開放する必要があります。
これを実現するため、画像ごとの `cudaArray*`, `cudaTextureObject_t`, `unsigned char*`, `cudaStream_t` を構造体にまとめて管理します。
最終的に処理が完了した後、Destroy()関数を実行することでまとめてリソースを解放します。

```
struct CudaImageResource {
    cudaStream_t stream = nullptr;
    cudaArray* cuArray = nullptr;
    cudaTextureObject_t texObj = 0;
    unsigned char* d_output = nullptr;
    cv::Mat output;

    void Destroy() { // デストラクタでもよい
        if (texObj) cudaDestroyTextureObject(texObj);
        if (cuArray) cudaFreeArray(cuArray);
        if (d_output) cudaFree(d_output);
        if (stream) cudaStreamDestroy(stream);
    }
};
```

### 2. 非同期化のポイント

* `cudaMemcpy2DToArrayAsync()`
* `grayscaleKernel<<<..., ..., 0, stream>>>`
* `cudaMemcpyAsync()`

を使うことで、各処理が非同期にGPUへ投入されます。

### 3. 処理の構成

以下の順に行いました：

* すべての画像について：メモリ確保とテクスチャオブジェクトの作成（非同期化できない処理）を**先に実行**
* その後、転送・カーネル・結果のコピーを**各streamで非同期に実行**
* 最後に全streamを `cudaStreamSynchronize()` で待機
* 結果を保存 & `Destroy()` により後始末

この構成により、非同期化できる処理だけがstreamに乗り、オーバーラップが最大化されます。

### 4. メモリの確保

私の実行環境で処理をしたところ、cudaArrayとしてメモリを確保した場合と、通常のデバイスメモリとして確保した場合で動作が違いました。
cudaArrayとしてメモリを確保した場合は、Streamの並列処理が行われず、すべての処理が同期的に実行されていました。
こちらについて詳しい原因はまだわかっていませんが、Textureを効率的に扱うため、cudaAraryは一般のcudaメモリと違う管理がされているらしいので、このあたりで動作の違いが出ているのかもしれません。
通常のcudaメモリとしてメモリを確保したあと、画像をメモリに詰めて処理をする方式だと、Streamの動作を確認できました。

## Nsightでの可視化結果

実行結果をNsightで可視化した結果を示します。

![nsight sequential](nsight_sequential.png)

<div style="text-align: center">
  <i>Sequentialの結果</i><br>
</div>

今回の実験では、同期的、非同期的処理それぞれにStreamを2つずつ使用しました。
最初の事件では、ストリームを分けてはいますが、すべての処理が重複せず、結果として同期的に動いているのと変わらない結果になっています。

![nsight streaming](nsight_streaming.png)

<div style="text-align: center">
  <i>Asyncの結果</i><br>
</div>

この画像では、**複数のメモリ転送やカーネル処理が並列で走っている**ことが明確に見て取れます。
特に緑色のバーと濃い青色の短いバーが部分的に重なっていることがわかります。
今回の実験ではカーネルの処理が短かったことや、すべての項目について最適化をしたわけではなかったため、一部のみ非同期に動作しています。
各種性能を最適化するとこういった部分の重複を増やすことができ、全体の処理を短縮できることになります。

---

## 学んだこと

* CUDAの非同期処理では、ストリームとリソースを一対一で保持する設計が有効
* `cudaMalloc` や `cudaCreateTextureObject` は非同期でないため、事前実行しておくことが重要
* Nsight Systemsでの可視化により、非同期処理の効果が直感的に確認できる
* リソース管理は構造体またはクラス化することで安全かつ明快に

---

Thanks for reading!  
👉 [GitHub Repo (cuda-examples)](https://github.com/yaikeda/cuda-examples/)
