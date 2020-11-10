#include "../utils/util.cuh"
using namespace akg_reduce;
using namespace std;

int main() {
  int items = 100;
  int bytes = items * sizeof(float);
  float *h_I, *d_I;
  h_I = (float *)malloc(bytes);

  GetGpuErr(cudaMalloc((void **)&d_I, bytes));
  // check if GetGpuErr can detect and return properly.
  GetGpuErr(cudaMemcpy((void *)d_I, (void *)h_I, bytes + 99, cudaMemcpyHostToDevice));
  GetGpuErr(cudaFree(d_I));
  free(h_I);

  return 0;
}
