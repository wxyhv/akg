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
#ifndef AKG_REDUCE_REDUCE_IMPL_H
#define AKG_REDUCE_REDUCE_IMPL_H
#include "./shared_reduce.cuh"

namespace akg_reduce {
  
// The implements for specific selected algorithms, 
// supports 1D & 2D (x-axis or y-axis) reduce.

template <typename T, typename ReduceOp, size_t blockSize_x>
__inline__ __device__ void AllReduce(ReduceOp op,       // the operator
                                     T *output,         // the addr of output, single value
                                     T *shared_array,   // the temp array in shared memory
                                     T acc              // aggregated value in current thread
) {
  HalvedReduce1D<T, ReduceOp, blockSize_x>(shared_array, &acc, op);
  if (threadIdx.x == 0) {
    output[0] = op(output[0], shared_array[0]);
  }
}

template <typename T, typename ReduceOp, size_t blockSize_x>
__inline__ __device__ void ReduceDirectionX(ReduceOp op,       // the operator
                                            T *output,         // the addr of output, single value
                                            T *shared_array,   // the temp array in shared memory
                                            T acc              // aggregated value in current thread
) {
  HalvedReduce2DX<T, ReduceOp, blockSize_x>(shared_array, &acc, op);

  if (threadIdx.x == 0) {
    output[0] = op(output[0], shared_array[threadIdx.y * blockDim.x]); // shared_array maybe a part of an array.
  }
}

template <typename T, typename ReduceOp, size_t blockSize_y>
__inline__ __device__ void ReduceDirectionY(ReduceOp op,       // the operator
                                            T *output,         // the addr of output, single value
                                            T *shared_array,   // the temp array in shared memory
                                            T acc,             // aggregated value in current thread
                                            int sharedmem_x    // shared memory size of x axis.
) {
  HalvedReduce2DY<T, ReduceOp, blockSize_y>(shared_array, &acc, op, sharedmem_x);
  if (threadIdx.y == 0) {
    output[0] = op(output[0], shared_array[threadIdx.x]);
  }
}

}  // namespace akg_reduce

#endif // AKG_REDUCE_REDUCE_IMPL_H