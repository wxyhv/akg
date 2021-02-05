- [AKG简述](#AKG简述)
- [硬件后端支持](#硬件后端支持)
- [构建](#构建)
    - [从MindSpore侧构建](#从MindSpore侧构建)
    - [独立构建](#独立构建)
- [贡献](#贡献)
- [版本说明](#版本说明)
- [许可证](#许可证)

[View English](./README.md)

## AKG简述
AKG(Auto Kernel Generator)对深度神经网络中的算子进行优化，并提供特定模式下的算子自动融合功能。AKG与MindSpore的图算融合功能协同工作，可提升在不同硬件后端上运行网络的性能。

AKG由基本的优化模块组成：规范化和自动调度。
- **规范化：** 为了解决polyhedral表达能力的局限性（只能处理静态的线性程序），需要首先对计算公式IR进行规范化。规范化模块中的优化主要包括自动运算符inline等。
- **自动调度：** 自动调度模块基于polyhedral技术，主要包括自动向量化、自动切分、依赖分析和数据搬移等。

## 硬件后端支持
当前`GPU`后端支持正在开发中。

## 构建

### 从MindSpore侧构建
详细细节请参考[MindSpore README.md](https://gitee.com/mindspore/mindspore/blob/master/README.md)。

### 独立构建
  ```
  bash build.sh
  ```

## 贡献

欢迎您的贡献，具体细节请参考[MindSpore贡献者Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)。

## 版本说明

版本说明详见[RELEASE](RELEASE.md).

## 许可证

[Apache License 2.0](LICENSE)
