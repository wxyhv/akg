- [What Is AKG?](#what-is-akg)
- [Hardware Backends Support](#hardware-backends-support)
- [Build](#build)
    - [Build With MindSpore](#build-with-mindspore)
    - [Build Standalone](#build-standalone)
- [Contributing](#contributing)
- [Release Notes](#release-notes)
- [License](#license)

[查看中文](./README_CN.md)

## What Is AKG
AKG(Auto Kernel Generator) is an optimizer for operators in Deep Learning Networks. It provides the ability to automatically fuse ops with specific patterns. AKG works with MindSpore-GraphKernel to improve the performance of networks running on different hardware backends.

AKG composes with basic optimization module, normalization and auto schedule.
- **normalization.** In order to solve the limitation in expression ability of polyhedral(which can only process static linear programs), the computation IR needs to be normalized first. The mainly optimization of normalization module includes auto-inline and so on.
- **auto schedule.** Base on polyhedral technology, the auto schedule module mainly have auto-vectorization, auto-tiling, dependency analysis and memory promotion.

## Hardware Backends Support
At present, GPU back-end is under development.

## Build

### Build With MindSpore
See [MindSpore README.md](https://gitee.com/mindspore/mindspore/blob/master/README.md) for details.

### Build Standalone
  ```
  bash build.sh
  ```

## Contributing

Welcome contributions. See [MindSpore Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md) for
more details.

## Release Notes

The release notes, see our [RELEASE](RELEASE.md).

## License

[Apache License 2.0](LICENSE)
