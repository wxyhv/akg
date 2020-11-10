#pragma once
#include "../utils/util.cuh"
#include "../operators/reduce_operators.cuh"

namespace akg_reduce {

template <typename OutputT, typename ReduceOp>
__device__ __forceinline__ void AtomicReturn1D(OutputT *shared_buf, OutputT *output, ReduceOp op) {
  if (threadIdx.x == 0) {
    AtomicOp<OutputT, op.identifier> atomic_op;
    atomic_op.Compute(&output[0], shared_buf[0]);
  }
}

template <typename OutputT, typename ReduceOp>
__device__ __forceinline__ void AtomicReturn2DX(OutputT *shared_buf, OutputT *output, ReduceOp op) {
  AtomicOp<OutputT, op.identifier> atomic_op;
  int shared_item = threadIdx.y * blockDim.x;
  if (threadIdx.x == 0) {
    atomic_op.Compute(&output[0], shared_buf[shared_item]);
  }
}

template <typename OutputT, typename ReduceOp>
__device__ __forceinline__ void AtomicReturn2DY(OutputT *shared_buf, OutputT *output, ReduceOp op) {
  AtomicOp<OutputT, op.identifier> atomic_op;
  int shared_item = threadIdx.x;
  if (threadIdx.y == 0) {
    atomic_op.Compute(&output[0], shared_buf[shared_item]);
  }
}

template <typename OutputT>
__device__ __forceinline__ void DirectReturn1D(OutputT *shared_buf, OutputT *output) {
  if (threadIdx.x == 0) {
    output[0] = shared_buf[0];
  }
}

template <typename OutputT>
__device__ __forceinline__ void DirectReturn2DX(OutputT *shared_buf, OutputT *output) {
  int shared_item = threadIdx.y * blockDim.x;
  if (threadIdx.x == 0) {
    output[0] = shared_buf[shared_item];
  }
}

template <typename OutputT>
__device__ __forceinline__ void DirectReturn2DY(OutputT *shared_buf, OutputT *output) {
  int shared_item = threadIdx.x;
  if (threadIdx.y == 0) {
    output[0] = shared_buf[shared_item];
  }
}

}  // namespace akg_reduce