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
