# openLab
研究室紹介でGPUを紹介するために使用するプログラムです．
Jetson Nanoで実行することを前提に作成されています．

**本プログラムを用いて行う一切の行為，被った損害・損失に対しては，一切の責任を負いかねます**

## 導入とビルド

### 1. 実行環境
このプログラムの実行に求められる環境は以下の通りです．
#### Jetson Nano
```
Software part of jetson-stats 4.2.3 - (c) 2023, Raffaello Bonghi
Model: NVIDIA Jetson Nano Developer Kit - Jetpack 4.6.4 [L4T 32.7.4]
NV Power Mode[0]: MAXN
Serial Number: [XXX Show with: jetson_release -s XXX]
Hardware:
 - P-Number: p3448-0000
 - Module: NVIDIA Jetson Nano (4 GB ram)
Platform:
 - Distribution: Ubuntu 18.04 Bionic Beaver
 - Release: 4.9.337-tegra
jtop:
 - Version: 4.2.3
 - Service: Active
Libraries:
 - CUDA: 10.2.300
 - cuDNN: 8.2.1.32
 - TensorRT: 8.2
 - VPI: 1.2.3
 - Vulkan: 1.2.70
 - OpenCV: 4.1.1 - with CUDA: YES
```

#### GCC
```
gcc (Ubuntu/Linaro 7.5.0-3ubuntu1~18.04) 7.5.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### 2. ビルド
ビルドする場合，OpenCVのCUDAを有効にする必要があります．
また，CUDAがインストールされているディレクトリへパスを通し，環境変数を設定する必要もあります．
以下の操作により，ビルドが可能です．
```
make
```
ビルドをクリーンアップするには，以下の操作を実行してください．
```
make clean
```

### 3. 実行
コマンドライン引数に**1**を指定することで，プログラムを実行できます．
なお，コマンドライン引数に**2**を指定した場合，特徴量分析のテストを実行します．
```
./main [1 | 2]
```

## 操作方法
プログラム実行後に特定の数字キーを入力することで，
カメラから取得した画像に画像処理を実行することが可能です．

|キー入力|実行される画像処理の内容|
|-|-|
|0|通常出力|
|1|ガンマ処理|
|2|ガンマ処理（openMP）|
|3|ガンマ処理（CUDA）|
|4|特徴量の分析|
|5|特徴量の分析（CUDA）|