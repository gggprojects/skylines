
#ifndef TEST_CU
#define TEST_CU

#include <cuda_runtime.h>


__global__ void ComputeChunkBruteForce(float *input_p, size_t begin, size_t end) {

}

extern "C" void ComputeTest() {
    int numBlocks = 1;
    dim3 threadsPerBlock(1, 1);
    //ComputeChunkBruteForce<<< numBlocks, threadsPerBlock >>>(nullptr, 0, 1);
}

#endif // !TEST_CU


