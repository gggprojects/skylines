#ifndef SKYLINES_QUERIES_ALGORITHMS_GPU_COMMON_CUH
#define SKYLINES_QUERIES_ALGORITHMS_GPU_COMMON_CUH

#include <vector>
#include <iostream>
#include <set>

#include <cuda_runtime.h>

#include "gpu/gpu_memory.hpp"
#include "queries/data/data_structures.hpp"
#include "queries/algorithms/algorithm.cuh"
#include "queries/algorithms/distance_type.hpp"

/*
Total amount of constant memory:    65536 bytes
sizeof(sl::queries::data::Point):   8 bytes
Max elements:                       65536 / 8 = 8192
*/
#define MAX_CONST_MEM_ELEMENTS 8192

__device__ inline bool NearestFunc(const float a, const float b) {
    return a <= b;
}

__device__ inline bool FurthestFunc(const float a, const float b) {
    return a >= b;
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


#endif
