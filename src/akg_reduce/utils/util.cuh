#pragma once
#include <iostream>
#include <cuda_fp16.h>
#include <string.h>

namespace akg_reduce {

const int ALL_REDUCE = 0;
const int REDUCE2D_X = 1;
const int REDUCE2D_Y = 2;
const int WARPSIZE = 32;

// Error detection functions
#ifndef GetGpuErr
#define GetGpuErr(e) \
  { GpuAssert((e), __FILE__, __LINE__); }
#endif

inline void GpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GET A GPU ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// Select the Type by if condition
template <bool IF, typename TrueType, typename FalseType>
struct Select {
  typedef TrueType Type;
};

template <typename TrueType, typename FalseType>
struct Select<false, TrueType, FalseType> {
  typedef FalseType Type;
};

/**
 * @brief the type transform function for test cases
 * 
 * @tparam T target type
 * @tparam Y original type
 * @param y  original value
 * @return   value with target type
 */
template <typename T, typename Y>
__host__ __device__ T TypeTransform(Y y) {
  if (sizeof(T) == 2) {
    return __float2half((double)y);
  } else if (sizeof(Y) == 2) {
    return (T)(__half2float(y));
  } else
    return (T)y;
}

__host__ __device__ bool IsPowOfTwo(unsigned int num) { return !(num & (num - 1));}

}  // namespace akg_reduce