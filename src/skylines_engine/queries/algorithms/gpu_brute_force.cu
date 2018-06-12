#include <vector>
#include <iostream>
#include <set>

#include <cuda_runtime.h>

#include "gpu/gpu_memory.hpp"
#include "queries/data/data_structures.hpp"
#include "queries/algorithms/algorithm.cuh"
#include "queries/algorithms/distance_type.hpp"
#include "queries/algorithms/gpu_common.cuh"

#define SHARED_MEM_ELEMENTS 1024

__constant__ sl::queries::data::Point device_input_q[MAX_CONST_MEM_ELEMENTS];

template<class Comparator>
__device__ void _ComputePartialSkyline(
    const sl::queries::data::WeightedPoint *input_p,
    size_t input_p_size,
    int input_q_size,
    Comparator comparator_function,
    sl::queries::data::Statistics *statistics,
    float *result) {

    __shared__ sl::queries::data::WeightedPoint shared_input_p[SHARED_MEM_ELEMENTS];

    int block_offset = blockIdx.x * blockDim.x; // we just have one dimension grids
    size_t global_pos = block_offset + threadIdx.x;

    sl::queries::data::WeightedPoint skyline_candidate(input_p[global_pos]);
    bool is_skyline = global_pos < input_p_size;

    sl::queries::data::Statistics thread_statistics;
    for (size_t current_input_p_pos = 0; current_input_p_pos < input_p_size; current_input_p_pos += SHARED_MEM_ELEMENTS) {
        //all threads in the block loads to shared
        shared_input_p[threadIdx.x] = input_p[threadIdx.x + current_input_p_pos];
        __syncthreads();

        if (is_skyline) {
            for (int i = 0; i < SHARED_MEM_ELEMENTS; i++) {
                if (current_input_p_pos + i != global_pos && current_input_p_pos + i < input_p_size) { // do not check against the same point
                    thread_statistics.num_comparisions_++;
                    if (sl::queries::algorithms::IsDominated(skyline_candidate, shared_input_p[i], device_input_q, input_q_size, comparator_function)) {
                        is_skyline = false;
                        break;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (is_skyline) {
        float max_distance = 0;
        for (int i = 0; i < input_q_size; i++) {
            float distance = skyline_candidate.SquaredDistance(device_input_q[i]);
            if (distance > max_distance) {
                max_distance = distance;
            }
        }
        result[global_pos] = max_distance;
    }
    else {
        result[global_pos] = -1;
    }

    atomicAdd(&statistics->num_comparisions_, thread_statistics.num_comparisions_);
}

__global__ void ComputePartialSkyline(
    const sl::queries::data::WeightedPoint *input_p,
    size_t input_p_size,
    int input_q_size,
    sl::queries::algorithms::DistanceType distance_type,
    sl::queries::data::Statistics *statistics,
    float *result) {

    switch (distance_type) {
    case sl::queries::algorithms::DistanceType::Nearest:
        _ComputePartialSkyline(input_p, input_p_size, input_q_size, NearestFunc, statistics, result);
        break;
    case sl::queries::algorithms::DistanceType::Furthest:
        _ComputePartialSkyline(input_p, input_p_size, input_q_size, FurthestFunc, statistics, result);
        break;
    default:
        break;
    }
}

extern "C" void ComputeGPUSkyline(
    const std::vector<sl::queries::data::WeightedPoint> &input_p,
    const std::vector<sl::queries::data::Point> &input_q,
    std::vector<sl::queries::data::WeightedPoint> *output,
    sl::queries::algorithms::DistanceType distance_type,
    size_t top_k,
    sl::queries::data::Statistics *stadistics_results) {

    sl::gpu::GPUStream gpu_stream;

    //copy to const memory the input Q
    cudaMemcpyToSymbolAsync(device_input_q, input_q.data(), sizeof(sl::queries::data::Point) * input_q.size(), 0, cudaMemcpyKind::cudaMemcpyHostToDevice, gpu_stream());

    size_t input_p_size = input_p.size();
    int input_q_size = static_cast<int>(input_q.size());

    size_t input_p_size_SHARED_MEM_SIZE_multiple = roundUp<size_t>(input_p.size(), SHARED_MEM_ELEMENTS);

    //copy to global memory the input P
    sl::gpu::GPUMemory<sl::queries::data::WeightedPoint> input_p_d(input_p_size_SHARED_MEM_SIZE_multiple);
    input_p_d.UploadToDeviceAsync(input_p, gpu_stream); //the final values maybe empty

                                                        //copy statistics
    sl::gpu::GPUMemory<sl::queries::data::Statistics> statistics_d(1);
    statistics_d.UploadToDeviceAsync(stadistics_results, 1, gpu_stream);

    sl::gpu::GPUMemory<float> result_d(input_p_size_SHARED_MEM_SIZE_multiple);
    /*
    MAX number of threads per MS is 2048.
    MAX number of threads per block 1024 => max blockDim.y = 1
    */
    dim3 threadsPerBlock(SHARED_MEM_ELEMENTS, 1);
    int total_numBlocks = static_cast<int>(divUp(input_p_size, static_cast<size_t>(threadsPerBlock.x * threadsPerBlock.y)));
    dim3 grid(total_numBlocks, 1);

    ComputePartialSkyline<<<grid, threadsPerBlock, 0, gpu_stream()>>>(input_p_d(), input_p_size, input_q_size, distance_type, statistics_d(), result_d());
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << cudaGetErrorString(e) << '\n';
    }
    std::vector<float> result(input_p_size);
    result_d.DownloadToHostAsync(result.data(), input_p_size, gpu_stream);
    statistics_d.DownloadToHostAsync(stadistics_results, gpu_stream);

    e = gpu_stream.Syncronize();
    if (e != cudaSuccess) {
        std::cout << cudaGetErrorString(e) << '\n';
    }

    std::set<sl::queries::algorithms::PointStatistics> points;
    float max_distance_in_set = 99999;
    for (size_t i = 0; i < result.size(); i++) {
        float distance = result[i];
        if (distance != -1) {
            //it's a skyline
            if (points.size() < top_k || distance < max_distance_in_set) {
                points.insert(sl::queries::algorithms::PointStatistics(input_p[i], std::make_pair(0.f, distance)));
                if (points.size() > top_k)
                    points.erase(points.begin());
                max_distance_in_set = points.begin()->s_.second;
            }
        }
    }

    for (const sl::queries::algorithms::PointStatistics &ps : points) {
        output->emplace_back(ps.wp_);
    }

    stadistics_results->output_size_ = output->size();
}





