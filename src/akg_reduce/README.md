#AKG_REDUCE

## 1. Brief introduction

AKG_REDUCE is a reduction algorithms library for NVIDIA-GPU (currently) in AKG. AKG_REDUCE aims to generate a better reduce kernels automatically by embedding template reduce code into the final CUDA code.

## 2. Features

- focuses on 1D and 2D reduction in fixed shared memory size. (thanks to the reduce support in previous pass, like axis fusing)
- uses "multi-blocks + atomic funs" strategies which can utilize the benefit of GPU traits.  
- provides an unified interface for cuda-code in codegen, which can satisfy diverse scenarios.

## 3. Usages

- The things you need to do are calling the function "AkgReduce" with kernel informations (reduce direction, all-reduce or 2D-reduce, reduction operator, etc) and "AkgAtomicReturn" for multi-block-reduce return. 
- See the "./test" for more informations.
