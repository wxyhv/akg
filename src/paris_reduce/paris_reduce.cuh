#pragma once
#include "../akg_reduce/reduce.cuh"

namespace paris_reduce {

template <typename T, typename ReduceOp, size_t BlockSizeReduce>
__inline__ __device__ void ParisAllReduce(ReduceOp op, T *output, T *shared_array, T acc) {
  shared_array[threadIdx.x] = acc;
  float4 *vec = ((float4 *)shared_array);
  int index;
  for (int delta = blockDim.x / 8; delta > 0; delta /= 2) {
    __syncthreads();
    if (threadIdx.x < delta) {
      index = threadIdx.x + delta;
      vec[threadIdx.x].x += vec[index].x;
      vec[threadIdx.x].y += vec[index].y;
      vec[threadIdx.x].z += vec[index].z;
      vec[threadIdx.x].w += vec[index].w;
    }
  }
  if (((int)threadIdx.x) == 0) {
    output[0] = vec[0].x + vec[0].y + vec[0].z + vec[0].w;
  }
}

template <typename T, typename ReduceOp, size_t BlockSizeReduce>
__inline__ __device__ void ParisReduceX(ReduceOp op, T *output, T *shared_array, T acc) {
  shared_array[threadIdx.x * blockDim.y + threadIdx.y] = acc;
  for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
    __syncthreads();
    if (threadIdx.x < delta) {
      shared_array[threadIdx.x * blockDim.y + threadIdx.y] +=
        shared_array[(threadIdx.x + delta) * blockDim.y + threadIdx.y];
    }
  }

  if (threadIdx.x == 0) {
    output[0] = shared_array[threadIdx.y];
  }
}

template <typename T, typename ReduceOp, size_t BlockSizeReduce>
__inline__ __device__ void ParisReduceY(ReduceOp op, T *output, T *shared_array, T acc) {
  shared_array[threadIdx.x * blockDim.y + threadIdx.y] = acc;
  for (int delta = blockDim.y / 2; delta > 0; delta /= 2) {
    __syncthreads();
    if (threadIdx.y < delta) {
      shared_array[threadIdx.x * blockDim.y + threadIdx.y] +=
        shared_array[threadIdx.x * blockDim.y + threadIdx.y + delta];
    }
  }

  if (threadIdx.y == 0) {
    output[0] = shared_array[threadIdx.x * blockDim.y];
  }
}

template <typename T, typename ReduceOp, size_t BlockSizeReduce, int ReduceType>
__inline__ __device__ void ParisReduce(
  ReduceOp op,         // the operator
  T *output_array,     // the addr of output in global/shared memory, single value
  T *shared_array,     // the temp array in shared memory
  T acc,               // aggregated value in current thread
  int sharedmem_x = 0  // shared memory size of x axis, especially used for reduce2D along y.
) {
  if (ReduceType == akg_reduce::ALL_REDUCE) {
    ParisAllReduce<T, ReduceOp, BlockSizeReduce>(op, output_array, shared_array, acc);
    return;
  }

  if (ReduceType == akg_reduce::REDUCE2D_X) {
    ParisReduceX<T, ReduceOp, BlockSizeReduce>(op, output_array, shared_array, acc);
  }

  if (ReduceType == akg_reduce::REDUCE2D_Y) {
    ParisReduceY<T, ReduceOp, BlockSizeReduce>(op, output_array, shared_array, acc, sharedmem_x);
  }
}

template <typename T, typename ReduceOp>
__device__ __forceinline__ void ParisReturn(T shared_result, T *output, ReduceOp op) {
  akg_reduce::AkgAtomicReturn<T, ReduceOp>(shared_result, output, op);
}

}  // namespace paris_reduce