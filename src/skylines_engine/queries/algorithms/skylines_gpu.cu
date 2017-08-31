#include <vector>
#include <iostream>

#include <cuda_runtime.h>

#include "queries/data/data_structures.hpp"
#include "gpu/gpu_memory.hpp"
#include "queries/algorithms/algorithm.cuh"

/*
MAX constant memory: 65536 bytes
Size of sl::queries::data::Point = 8 bytes
MAX input_q 65536 / 8 = 8192
*/
__constant__ sl::queries::data::Point device_input_q[8192];

/*
MAX shared memory per block = 49152 bytes
Size of sl::queries::data::WeightedPoint = 12 bytes
MAX shared_input_p 49152 / 12 = 4096
*/

#define SHARED_MEM_SIZE 32

__global__ void ComputePartialSkyline(
    const sl::queries::data::WeightedPoint *input_p, 
    size_t input_p_size,
    int input_q_size,
    unsigned int *result) {

    __shared__ sl::queries::data::WeightedPoint shared_input_p[SHARED_MEM_SIZE];

    int block_size = blockDim.x * blockDim.y;
    int block_offset = blockIdx.x * block_size; // we just have one dimension grids
    int block_pos = blockDim.x * threadIdx.y + threadIdx.x;
    size_t global_pos = block_offset + block_pos;

    const sl::queries::data::WeightedPoint &skyline_candidate = input_p[global_pos];
    bool is_skyline = global_pos < input_p_size;

    for (size_t current_input_p_pos = 0; current_input_p_pos < input_p_size; current_input_p_pos += SHARED_MEM_SIZE) {
        //threads from the first line load to shared memory
        if (threadIdx.y == 0) {
            shared_input_p[threadIdx.x] = input_p[threadIdx.x + current_input_p_pos];
        }
        __syncthreads();

        if (is_skyline) {
            //#pragma unroll SHARED_MEM_SIZE
            for (int i = 0; i < SHARED_MEM_SIZE; i++) {
                if (current_input_p_pos + i != global_pos) { // do not check against the same point
                    if (IsDominated_impl(skyline_candidate, shared_input_p[i], device_input_q, input_q_size)) {
                        is_skyline = false;
                        break;
                    }
                }
            }
        }
        __syncthreads();
    }

    result[global_pos] = is_skyline ? 1 : 0;
}

template<typename T>
T inline divUp(T a, T b) {
    return (a + b - 1) / b;
}

template<typename T>
T roundUp(T numToRound, T multiple)
{
    if (multiple == 0)
        return numToRound;

    T remainder = numToRound % multiple;
    if (remainder == 0)
        return numToRound;

    return numToRound + multiple - remainder;
}

extern "C" void ComputeGPUSkyline(
    const std::vector<sl::queries::data::WeightedPoint> &input_p,
    const std::vector<sl::queries::data::Point> &input_q,
    std::vector<sl::queries::data::WeightedPoint> *output) {

    sl::gpu::GPUStream gpu_stream;

    //copy to const memory the input Q
    cudaMemcpyToSymbolAsync(device_input_q, input_q.data(), sizeof(float2) * input_q.size(), 0, cudaMemcpyKind::cudaMemcpyHostToDevice, gpu_stream());

    size_t input_p_size = input_p.size();
    int input_q_size = static_cast<int>(input_q.size());

    size_t input_p_size_32_multiple = roundUp<size_t>(input_p.size(), SHARED_MEM_SIZE);

    //copy to global memory the input P
    sl::gpu::GPUMemory<sl::queries::data::WeightedPoint> input_p_d(input_p_size_32_multiple);
    input_p_d.UploadToDeviceAsync(input_p, gpu_stream); //the final values maybe empty
    sl::gpu::GPUMemory<unsigned int> result_d(input_p_size_32_multiple);

    /*
    MAX number of threads per MS is 2048.
    MAX number of threads per block 1024 => max blockDim.y = 32
    */
    size_t num_rows = divUp<size_t>(input_p_size_32_multiple, SHARED_MEM_SIZE);
    int num_rows_per_block = num_rows < 32 ? static_cast<int>(num_rows) : 32;
    dim3 threadsPerBlock(32, num_rows_per_block);

    int numBlocks = static_cast<int>(divUp(input_p_size, static_cast<size_t>(threadsPerBlock.x * threadsPerBlock.y)));
    ComputePartialSkyline<<< numBlocks, threadsPerBlock, SHARED_MEM_SIZE, gpu_stream() >>>(input_p_d(), input_p_size, input_q_size, result_d());

    std::vector<unsigned int> result(input_p_size);
    result_d.DownloadToHostAsync(result.data(), result.size(), gpu_stream);
    gpu_stream.Syncronize();

    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] == 1) {
            output->push_back(input_p[i]);
        }
    }
}



