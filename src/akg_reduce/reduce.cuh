#pragma once
#include "./utils/util.cuh"
#include "./algorithm/reduce_impl.cuh"
#include "./operators/reduce_operators.cuh"

namespace akg_reduce {
/**
 * main functions of reduce module
 */

/**
 * @brief this function pick up the proper strategies for different kinds of cases automatically.

 * @tparam T                  dtype: half, float, double, int, signed char, bool;
 * @tparam ReduceOp           operators for reduce: SumOp, MaxOp, MinOp, AndOp, OrOp;
 * @tparam BlockSizeReduce    the length of reduce axis
 * @tparam ReduceType         types of reduce: ALL_REDUCE(0), REDUCE2D_X(1), REDUCE2D_Y(2); 
 */
template <typename T, typename ReduceOp, size_t BlockSizeReduce, int ReduceType>
__inline__ __device__ void AkgReduce(
  ReduceOp op,           // the operator
  // int reduce_direction,  // 0 for direction x; 1 for direction y
  T *output_array,      // the addr of output in global/shared memory, single value
  T *shared_array,       // the temp array in shared memory
  T acc,                 // aggregated value in current thread
  int sharedmem_x = 0    // shared memory size of x axis, especially used for reduce2D along y.
) {
  // all-reduce
  if (ReduceType == ALL_REDUCE) {
    AllReduce<T, ReduceOp, BlockSizeReduce>(op, output_array, shared_array, acc);
    return;
  }

  // reduce data from direction x
  if (ReduceType == REDUCE2D_X) {
    ReduceDirectionX<T, ReduceOp, BlockSizeReduce>(op, output_array, shared_array, acc);
  }

  // reduce data from direction y
  if (ReduceType == REDUCE2D_Y){
    ReduceDirectionY<T, ReduceOp, BlockSizeReduce>(op, output_array, shared_array, acc, sharedmem_x);
  }
}

/**
 * @brief the atomic return function, from shared memory to global memory
 */
template <typename OutputT, typename ReduceOp>
__device__ __forceinline__ void AkgAtomicReturn(OutputT shared_result, OutputT *output, ReduceOp op) {
  AtomicOp<OutputT, op.identifier> atomic_op;
  atomic_op.Compute(&output[0], shared_result);
}

}  // namespace akg_reduce
