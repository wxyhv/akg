/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef AKG_REDUCE_SHARED_REDUCE_H
#define AKG_REDUCE_SHARED_REDUCE_H
#include "../utils/util.cuh"
#include "../operators/reduce_operators.cuh"

/*********************************************************
 * Algorithms of reduction computation in shared memory.
 * ********************************************************/
namespace akg_reduce {

/**
 * \brief Reduction within Warp. Using unroll strategy.
 *
 * \par
 * - When blockDim.x is power of two, using completely unrolled strategy within warp.
 * - Supports 1D or 2D reduction computation. The reduction direction is along x-axis.
 *
 * \tparam T                 Output type of reduction.
 * \tparam ReduceOp          Reduce operator type.
 * \tparam blockSize_x       block size along x axis.
 **/
template <typename T, typename ReduceOp, size_t blockSize_x>
__device__ __forceinline__ void WarpReduceUnroll(volatile T *shared_buf,  // shared memory buffer.
                                                 T *reg_buf,              // register buffer.
                                                 ReduceOp op              // reduce operator.
) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  if (blockSize_x >= 64) {
    shared_buf[tid] = op(static_cast<T>(shared_buf[tid]), static_cast<T>(shared_buf[tid + 32]));
  }
  if (blockSize_x >= 32) {
    if (threadIdx.x < 16) shared_buf[tid] = op(static_cast<T>(shared_buf[tid]), static_cast<T>(shared_buf[tid + 16]));
  }
  if (blockSize_x >= 16) {
    if (threadIdx.x < 8) shared_buf[tid] = op(static_cast<T>(shared_buf[tid]), static_cast<T>(shared_buf[tid + 8]));
  }
  if (blockSize_x >= 8) {
    if (threadIdx.x < 4) shared_buf[tid] = op(static_cast<T>(shared_buf[tid]), static_cast<T>(shared_buf[tid + 4]));
  }
  if (blockSize_x >= 4) {
    if (threadIdx.x < 2) shared_buf[tid] = op(static_cast<T>(shared_buf[tid]), static_cast<T>(shared_buf[tid + 2]));
  }
  if (blockSize_x >= 2) {
    if (threadIdx.x < 1) shared_buf[tid] = op(static_cast<T>(shared_buf[tid]), static_cast<T>(shared_buf[tid + 1]));
  }
}

/**
 * \brief Reduction within Warp. Without unroll strategy.
 *
 * \par
 * - When blockDim.x is not the power of two, halving reduce_align to one with iteration for reduction.
 * - Supports 1D or 2D reduction computation. The reduction direction is along x-axis.
 * - The first tid upper bound is reduce_align_upper, which is less than reduce_align. The operation avoids to access
 *illegal memory.
 *
 * \tparam T                 Output type of reduction.
 * \tparam ReduceOp          Reduce operator type.
 *
 **/
template <typename T, typename ReduceOp>
__device__ __forceinline__ void WarpReduce(volatile T *shared_buf,  // shared memory buffer
                                           T *reg_buf,              // register buffer
                                           ReduceOp op,             // reduce operator
                                           int reduce_align,        // reduce align, which is power of two
                                           int reduce_align_upper)  // less than reduce_align, depending on blockDim.x
{
  int tid = threadIdx.y * blockDim.x + threadIdx.x;

  if (threadIdx.x < reduce_align_upper) {
    shared_buf[tid] = op(static_cast<T>(shared_buf[tid]), static_cast<T>(shared_buf[tid + reduce_align]));
  }
  reduce_align = reduce_align >> 1;

  if (threadIdx.x < reduce_align) {
    while (reduce_align >= 1) {
      shared_buf[tid] = op(static_cast<T>(shared_buf[tid]), static_cast<T>(shared_buf[tid + reduce_align]));
      reduce_align = reduce_align >> 1;
    }
  }
}

/**
 * \brief Reduction along x axis with unroll strategy.
 *
 * \par
 * - when blockDim.x is power of two, completely unrolling the reduction.
 * - support 1D or 2D reduction.
 *
 * \tparam T                 Output type of reduction.
 * \tparam ReduceOp          Reduce operator type.
 * \tparam blockSize_x       block size along x axis.
 *
 **/
template <typename T, typename ReduceOp, size_t blockSize_x>
__device__ __forceinline__ void ReduceXUnroll(T *shared_buf,  // shared memory buffer.
                                              T *reg_buf,     // register buffer.
                                              ReduceOp op     // reduce operator.
) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  if (blockSize_x >= 1024) {
    if (threadIdx.x < 512) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 512]);
    }
    __syncthreads();
  }
  if (blockSize_x >= 512) {
    if (threadIdx.x < 256) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 256]);
    }
    __syncthreads();
  }
  if (blockSize_x >= 256) {
    if (threadIdx.x < 128) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 128]);
    }
    __syncthreads();
  }
  if (blockSize_x >= 128) {
    if (threadIdx.x < 64) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 64]);
    }
    __syncthreads();
  }
  if (threadIdx.x < 32) {
    WarpReduceUnroll<T, ReduceOp, blockSize_x>(shared_buf, reg_buf, op);
  }
  __syncthreads();
}

/**
 * \brief Reduction along y axis with unroll strategy.
 *
 * \par
 * - when blockDim.y is power of two, completely unrolling the reduction.
 * - support 2D reduction.
 * - shared memory size along x axis could be set as proper odd to avoid bank conflicts.
 *
 * \tparam T                 Output type of reduction.
 * \tparam ReduceOp          Reduce operator type.
 * \tparam blockSize_y       block size along y axis.
 *
 **/
template <typename T, typename ReduceOp, size_t blockSize_y>
__device__ __forceinline__ void ReduceYUnroll(T *shared_buf,   // shared memory buffer.
                                              T *reg_buf,      // register buffer.
                                              ReduceOp op,     // reduce operator.
                                              int sharedmem_x  // shared memory size of x axis.
) {
  int tid = threadIdx.y * sharedmem_x + threadIdx.x;
  if (blockSize_y >= 1024) {
    if (threadIdx.y < 512) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 512 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (blockSize_y >= 512) {
    if (threadIdx.y < 256) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 256 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (blockSize_y >= 256) {
    if (threadIdx.y < 128) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 128 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (blockSize_y >= 128) {
    if (threadIdx.y < 64) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 64 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (blockSize_y >= 64) {
    if (threadIdx.y < 32) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 32 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (blockSize_y >= 32) {
    if (threadIdx.y < 16) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 16 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (blockSize_y >= 16) {
    if (threadIdx.y < 8) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 8 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (blockSize_y >= 8) {
    if (threadIdx.y < 4) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 4 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (blockSize_y >= 4) {
    if (threadIdx.y < 2) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 2 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (blockSize_y >= 2) {
    if (threadIdx.y < 1) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 1 * sharedmem_x]);
    }
    __syncthreads();
  }
}

/**
 * \brief All-Reduce computation with halved reduce algorithm.
 *
 * \par
 * - support single-block reduction and multi-block reduction.
 * - support 1D reduction.
 *
 * \tparam T                 Output type of reduction.
 * \tparam ReduceOp          Reduce operator type.
 * \tparam blockSize_x       block size along x axis.
 *
 **/
template <typename T, typename ReduceOp, size_t blockSize_x>
__device__ __forceinline__ void HalvedReduce1D(T *shared_buf, T *reg_buf, ReduceOp op) {
  int reduce_extent = blockDim.x;

  // load data to shared memory.
  shared_buf[threadIdx.x] = reg_buf[0];
  __syncthreads();

  // if reduce extent is 1, just return.
  if (reduce_extent == 1) {
    return;
  }

  int tid = threadIdx.x;
  if (IsPowOfTwo(blockSize_x)) {
    // Using unroll strategy.
    ReduceXUnroll<T, ReduceOp, blockSize_x>(shared_buf, reg_buf, op);
  } else {
    // Using iteration to compute.
    int reduce_align = 1;

    while (reduce_extent > reduce_align) {
      // reduce_extent is power of two and large than reduce_extent.
      reduce_align = reduce_align << 1;
    }

    reduce_align = reduce_align >> 1;
    // The first iteration cannot access whole reduce_align.
    int reduce_align_upper = reduce_extent - reduce_align;

    while (reduce_align > WARPSIZE) {
      if (threadIdx.x < reduce_align_upper) {
        int tid_new = tid + reduce_align;
        shared_buf[tid] = op(static_cast<T>(shared_buf[tid]), static_cast<T>(shared_buf[tid_new]));
      }
      __syncthreads();
      reduce_align = reduce_align >> 1;
      reduce_align_upper = reduce_align;
    }
    // Do warp reduce.
    WarpReduce<T, ReduceOp>(shared_buf, reg_buf, op, reduce_align, reduce_align_upper);
    __syncthreads();

  }
}

/**
 * \brief Part-of-Reduce computation with halved reduce algorithm.
 *
 * \par
 * - Reduction direction is x axis.
 * - support single-block reduction and multi-block reduction.
 * - support 2D reduction.
 *
 * \tparam T                 Output type of reduction.
 * \tparam ReduceOp          Reduce operator type.
 * \tparam blockSize_x       block size along x axis.
 *
 **/

template <typename T, typename ReduceOp, size_t blockSize_x>
__device__ __forceinline__ void HalvedReduce2DX(T *shared_buf, T *reg_buf, ReduceOp op) {
  int reduce_extent = blockDim.x;

  // load data to shared memory.
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  shared_buf[tid] = reg_buf[0];
  __syncthreads();

  if (reduce_extent == 1) {
    return;
  }

  if (IsPowOfTwo(blockSize_x)) {
    ReduceXUnroll<T, ReduceOp, blockSize_x>(shared_buf, reg_buf, op);
  } else {
    int reduce_align = 1;
    while (reduce_extent > reduce_align) {
      reduce_align = reduce_align << 1;
    }
    reduce_align = reduce_align >> 1;

    int reduce_align_upper = reduce_extent - reduce_align;
    while (reduce_align > WARPSIZE) {
      if (threadIdx.x < reduce_align_upper) {
        int tid_new = tid + reduce_align;
        shared_buf[tid] = op(shared_buf[tid], shared_buf[tid_new]);
      }
      __syncthreads();
      reduce_align = reduce_align >> 1;
      reduce_align_upper = reduce_align;
    }
    WarpReduce<T, ReduceOp>(shared_buf, reg_buf, op, reduce_align, reduce_align_upper);
  __syncthreads();
  }
}

/**
 * \brief Part-of-Reduce computation with halved reduce algorithm.
 *
 * \par
 * - Reduction direction is y axis.
 * - support single-block reduction and multi-block reduction.
 * - support 2D reduction.
 * - support set sharedmem_x as a proper odd to avoid bank conflicts.
 *
 * \tparam T                 Output type of reduction.
 * \tparam ReduceOp          Reduce operator type.
 * \tparam blockSize_y       block size along y axis.
 *
 **/

template <typename T, typename ReduceOp, size_t blockSize_y>
__device__ __forceinline__ void HalvedReduce2DY(T *shared_buf, T *reg_buf, ReduceOp op, int sharedmem_x) {
  int reduce_extent = blockDim.y;

  // Load data to shared memory.
  // Note： NO size CHECK here. It depends on the set value for reg_variable.
  int tid = threadIdx.y * sharedmem_x + threadIdx.x;
  shared_buf[tid] = reg_buf[0];
  __syncthreads();

  if (reduce_extent == 1) {
    return;
  }

  if (IsPowOfTwo(blockSize_y)) {
    ReduceYUnroll<T, ReduceOp, blockSize_y>(shared_buf, reg_buf, op, sharedmem_x);
  } else {
    int reduce_align = 1;

    while (reduce_extent > reduce_align) {
      reduce_align = reduce_align << 1;
    }
    reduce_align = reduce_align >> 1;

    int reduce_align_upper = reduce_extent - reduce_align;

    while (reduce_align >= 1) {
      if (threadIdx.y < reduce_align_upper) {
        int tid_new = tid + reduce_align * sharedmem_x;
        shared_buf[tid] = op(shared_buf[tid], shared_buf[tid_new]);
      }
      __syncthreads();
      reduce_align = reduce_align >> 1;
      reduce_align_upper = reduce_align;
    }
  }
}
}  // namespace akg_reduce

#endif // AKG_REDUCE_SHARED_REDUCE_H