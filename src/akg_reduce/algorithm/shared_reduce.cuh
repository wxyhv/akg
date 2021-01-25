/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
 * - When BlockDimX is power of two, using completely unrolled strategy within warp.
 * - Supports 1D or 2D reduction computation. The reduction direction is along x-axis.
 * - Exclude cases when T == T, since shfl.sync funcs only support 16 bits,
 * - 32 bits and 64 bits.
 *
 * \tparam ReduceOp          Reduce operator type.
 * \tparam BlockDimX       block size along x axis.
 * \tparam T                 Output type of reduction.
 **/
template <typename ReduceOp, size_t BlockDimX, typename T>
__device__ __forceinline__ void WarpReduceUnroll(T *shared_buf,     // shared memory buffer.
                                                 ReduceOp op,       // reduce operator.
                                                 const int tx = 0,  // real tx
                                                 const int ty = 0   // real ty
) {
  const int tid = ty * BlockDimX + tx;
  T local_sum = shared_buf[tid];
  if (BlockDimX >= 32) {
    local_sum = op(local_sum, __shfl_down_sync(0xFFFFFFFF, local_sum, 16));
  }
  if (BlockDimX >= 16) {
    local_sum = op(local_sum, __shfl_down_sync(0xFFFFFFFF, local_sum, 8));
  }
  if (BlockDimX >= 8) {
    local_sum = op(local_sum, __shfl_down_sync(0xFFFFFFFF, local_sum, 4));
  }
  if (BlockDimX >= 4) {
    local_sum = op(local_sum, __shfl_down_sync(0xFFFFFFFF, local_sum, 2));
  }
  if (BlockDimX >= 2) {
    local_sum = op(local_sum, __shfl_down_sync(0xFFFFFFFF, local_sum, 1));
  }
  if (tx == 0) {
    shared_buf[tid] = local_sum;
  }
}

/**
 * \brief Reduction within Warp for one btye dtype. Using unroll strategy.
 *
 * \par
 * - When BlockDimX is power of two, using completely unrolled strategy within warp.
 * - Supports 1D or 2D reduction computation. The reduction direction is along x-axis.
 *
 * \tparam ReduceOp          Reduce operator type.
 * \tparam BlockDimX       block size along x axis.
 **/
template <typename ReduceOp, size_t BlockDimX, typename T>
__device__ __forceinline__ void WarpReduceUnrollOneByte(T *shared_buf,  // shared memory buffer.
                                                 ReduceOp op,              // reduce operator.
                                                 const int tx = 0,         // real threadIdx.x
                                                 const int ty = 0          // real threadIdx.y
) {
  const int tid = ty * BlockDimX + tx;
  if (BlockDimX >= 32) {
    if (tx < 16)
      ((volatile T *)shared_buf)[tid] =
        op(((volatile T *)shared_buf)[tid], ((volatile T *)shared_buf)[tid + 16]);
  }
  if (BlockDimX >= 16) {
    if (tx < 8)
      ((volatile T *)shared_buf)[tid] =
        op(((volatile T *)shared_buf)[tid], ((volatile T *)shared_buf)[tid + 8]);
  }
  if (BlockDimX >= 8) {
    if (tx < 4)
      ((volatile T *)shared_buf)[tid] =
        op(((volatile T *)shared_buf)[tid], ((volatile T *)shared_buf)[tid + 4]);
  }
  if (BlockDimX >= 4) {
    if (tx < 2)
      ((volatile T *)shared_buf)[tid] =
        op(((volatile T *)shared_buf)[tid], ((volatile T *)shared_buf)[tid + 2]);
  }
  if (BlockDimX >= 2) {
    if (tx < 1)
      ((volatile T *)shared_buf)[tid] =
        op(((volatile T *)shared_buf)[tid], ((volatile T *)shared_buf)[tid + 1]);
  }
}

/**
 * \brief Reduction within Warp. Without unroll strategy.
 *
 * \par
 * - When BlockDimX is not the power of two, halving reduce_align to one with iteration for reduction.
 * - Supports 1D or 2D reduction computation. The reduction direction is along x-axis.
 * - The first tid upper bound is reduce_align_upper, which is less than reduce_align. The operation avoids to access
 *illegal memory.
 *
 * \tparam ReduceOp          Reduce operator type.
 * \tparam T                 Output type of reduction.
 *
 **/
template <typename ReduceOp, size_t BlockDimX, typename T>
__device__ __forceinline__ void WarpReduce(
  volatile T *shared_buf,         // shared memory buffer
  ReduceOp op,                    // reduce operator
  int reduce_align,         // reduce align, which is power of two
  const int reduce_align_upper,  // less than reduce_align, depending on BlockDimX
  const int tx = 0,               // real threadIdx.x
  const int ty = 0                // real threadIdx.y
){
  int tid = ty * BlockDimX + tx;

  if (tx < reduce_align_upper) {
    shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + reduce_align]);
  }
  reduce_align = reduce_align >> 1;

  if (tx < reduce_align) {
    while (reduce_align >= 1) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + reduce_align]);
      reduce_align = reduce_align >> 1;
    }
  }
}

/**
 * \brief Reduction along x axis with unroll strategy.
 *
 * \par
 * - when BlockDimX is power of two, completely unrolling the reduction.
 * - support 1D or 2D reduction.
 *
 * \tparam ReduceOp          Reduce operator type.
 * \tparam BlockDimX         Real block size along x axis.
 * \tparam T                 Output type of reduction.
 *
 **/
template <typename ReduceOp, size_t BlockDimX, typename T>
__device__ __forceinline__ void ReduceXUnroll(T *shared_buf,     // shared memory buffer.
                                              ReduceOp op,       // reduce operator.
                                              const int tx = 0,  // real threadIdx.x
                                              const int ty = 0   // real threadIdx.y
) {
  const int tid = ty * BlockDimX + tx;
  if (BlockDimX >= 1024) {
    if (tx < 512) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 512]);
    }
    __syncthreads();
  }
  if (BlockDimX >= 512) {
    if (tx < 256) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 256]);
    }
    __syncthreads();
  }
  if (BlockDimX >= 256) {
    if (tx < 128) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 128]);
    }
    __syncthreads();
  }
  if (BlockDimX >= 128) {
    if (tx < 64) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 64]);
    }
    __syncthreads();
  }
  if (BlockDimX >= 64) {
    if (tx < 32) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 32]);
    }
  }
  if (tx < 32) {
    // choose proper algorithm for different dtype.
    if (sizeof(T) == 1) {
      WarpReduceUnrollOneByte<ReduceOp, BlockDimX>(shared_buf, op, tx, ty);
    } else {
      WarpReduceUnroll<ReduceOp, BlockDimX>(shared_buf, op, tx, ty);
    }
  }
  __syncthreads();
}

/**
 * \brief Reduction along y axis with unroll strategy.
 *
 * \par
 * - when BlockDimY is power of two, completely unrolling the reduction.
 * - support 2D reduction.
 * - shared memory size along x axis could be set as proper odd to avoid bank conflicts.
 *
 * \tparam T                 Output type of reduction.
 * \tparam ReduceOp          Reduce operator type.
 * \tparam BlockDimX       block size along y axis.
 * \tparam BlockDimY       block size along y axis.
 *
 **/
template <typename ReduceOp, size_t BlockDimX, size_t BlockDimY, typename T>
__device__ __forceinline__ void ReduceYUnroll(T *shared_buf,          // shared memory buffer.
                                              ReduceOp op,            // reduce operator.
                                              const int sharedmem_x,  // shared memory size of x axis.
                                              const int tx = 0,       // real threadIdx.x
                                              const int ty = 0        // real threadIdx.y
) {
  const int tid = ty * sharedmem_x + tx;
  if (BlockDimY >= 1024) {
    if (ty < 512) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 512 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 512) {
    if (ty < 256) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 256 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 256) {
    if (ty < 128) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 128 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 128) {
    if (ty < 64) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 64 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 64) {
    if (ty < 32) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 32 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 32) {
    if (ty < 16) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 16 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 16) {
    if (ty < 8) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 8 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 8) {
    if (ty < 4) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 4 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 4) {
    if (ty < 2) {
      shared_buf[tid] = op(shared_buf[tid], shared_buf[tid + 2 * sharedmem_x]);
    }
    __syncthreads();
  }
  if (BlockDimY >= 2) {
    if (ty < 1) {
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
 * \tparam ReduceOp          Reduce operator type.
 * \tparam BlockDimX       block size along x axis.
 * \tparam T                 Output type of reduction.
 *
 **/
template <typename ReduceOp, size_t BlockDimX, typename T>
__device__ __forceinline__ void HalvedReduce1D(T *shared_buf, 
                                               const T local_acc, 
                                               ReduceOp op, 
                                               const int tx = 0,
                                               const int ty = 0) {

                                                
  constexpr int reduce_extent = BlockDimX;
  // load data to shared memory.
  shared_buf[tx] = local_acc;
  __syncthreads();

  // if reduce extent is 1, just return.
  if (BlockDimX == 1) {
    return;
  }

  const int tid = tx;
  if (IsPowOfTwo(BlockDimX)) {
    // Using unroll strategy.
    ReduceXUnroll<ReduceOp, BlockDimX>(shared_buf, op, tx, ty);
  } else {
    // Using iteration to compute.
    
    int reduce_align = 1;

    while (BlockDimX > reduce_align) {
      reduce_align = reduce_align << 1;
    }

    reduce_align = reduce_align >> 1;
    // The first iteration cannot access whole reduce_align.
    int reduce_align_upper = reduce_extent - reduce_align;

    while (reduce_align > WARPSIZE) {
      if (tx < reduce_align_upper) {
        int tid_new = tid + reduce_align;
        shared_buf[tid] = op(shared_buf[tid], shared_buf[tid_new]);
      }
      __syncthreads();
      reduce_align = reduce_align >> 1;
      reduce_align_upper = reduce_align;
    }
    // Do warp reduce.
    WarpReduce<ReduceOp, BlockDimX>(shared_buf, op, reduce_align, reduce_align_upper, tx, ty);
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
 * \tparam ReduceOp          Reduce operator type.
 * \tparam BlockDimX       block size along x axis.
 * \tparam T                 Output type of reduction.
 *
 **/

template <typename ReduceOp, size_t BlockDimX, typename T>
__device__ __forceinline__ void HalvedReduce2DX(T *shared_buf, 
                                                const T local_acc, 
                                                ReduceOp op, 
                                                const int tx = 0,
                                                const int ty = 0) {
  const int tid = ty * BlockDimX + tx;

  // load data to shared memory.
  shared_buf[tid] = local_acc;
  __syncthreads();

  if (BlockDimX == 1) {
    return;
  }

  if (IsPowOfTwo(BlockDimX)) {
    ReduceXUnroll<ReduceOp, BlockDimX>(shared_buf, op, tx, ty);
  } else {
    int reduce_align = 1;
    while (BlockDimX > reduce_align) {
      reduce_align = reduce_align << 1;
    }
    reduce_align = reduce_align >> 1;

    int reduce_align_upper = BlockDimX - reduce_align;
    while (reduce_align > WARPSIZE) {
      if (tx < reduce_align_upper) {
        const int tid_new = tid + reduce_align;
        shared_buf[tid] = op(shared_buf[tid], shared_buf[tid_new]);
      }
      __syncthreads();
      reduce_align = reduce_align >> 1;
      reduce_align_upper = reduce_align;
    }
    WarpReduce<ReduceOp, BlockDimX>(shared_buf, op, reduce_align, reduce_align_upper, tx, ty);
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
 * \tparam BlockDimX       real blockDim.x
 * \tparam BlockDimY       real blockDim.y
 *
 **/

template <typename ReduceOp, size_t BlockDimX, size_t BlockDimY, typename T>
__device__ __forceinline__ void HalvedReduce2DY(T *shared_buf, 
                                                const T local_acc, 
                                                ReduceOp op, 
                                                const int sharedmem_x,
                                                const int tx = 0, 
                                                const int ty = 0) {
  constexpr int reduce_extent = BlockDimY;

  // Load data to shared memory.
  // Note： NO size CHECK here. It depends on the set value for reg_variable.
  const int tid = ty * sharedmem_x + tx;
  shared_buf[tid] = local_acc;
  __syncthreads();

  if (BlockDimY == 1) {
    return;
  }

  if (IsPowOfTwo(BlockDimY)) {
    ReduceYUnroll<ReduceOp, BlockDimX, BlockDimY>(shared_buf, op, sharedmem_x, tx, ty);
  } else {
    int reduce_align = 1;

    while (reduce_extent > reduce_align) {
      reduce_align = reduce_align << 1;
    }
    reduce_align = reduce_align >> 1;

    int reduce_align_upper = reduce_extent - reduce_align;

    while (reduce_align >= 1) {
      if (ty < reduce_align_upper) {
        const int tid_new = tid + reduce_align * sharedmem_x;
        shared_buf[tid] = op(shared_buf[tid], shared_buf[tid_new]);
      }
      __syncthreads();
      reduce_align = reduce_align >> 1;
      reduce_align_upper = reduce_align;
    }
  }
}
}  // namespace akg_reduce

#endif  // AKG_REDUCE_SHARED_REDUCE_H