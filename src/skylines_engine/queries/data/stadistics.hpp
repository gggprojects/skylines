#ifndef SKYLINES_QUERIES_DATA_STADISTICS_HPP
#define SKYLINES_QUERIES_DATA_STADISTICS_HPP

#include <cuda_runtime.h>
#include "queries/data/data_structures.hpp"

namespace sl { namespace queries { namespace data {
    struct __align__(32) Statistics {

        __host__ __device__ Statistics() : num_comparisions_(0), output_size_(0) {
        }

        Statistics operator+=(const Statistics &other) {
            num_comparisions_ += other.num_comparisions_;
            output_size_ += other.output_size_;
            max_distance_ += other.max_distance_;
            min_distance_ += other.min_distance_;
            return *this;
        }

        size_t num_comparisions_;
        size_t output_size_;
        float max_distance_;
        float min_distance_;
    };
}}}

__host__ __device__ inline void ComputeStatistics(
    float a_distance, float b_distance,
    sl::queries::data::Statistics *statistics) {
    //number of comparisons
    statistics->num_comparisions_++;

    //the max distance
    float max_distance = fmaxf(a_distance, b_distance);
    if (max_distance > statistics->max_distance_) {
        statistics->max_distance_ = max_distance;
    }

    //the min distance
    float min_distance = fminf(a_distance, b_distance);
    if (min_distance < statistics->max_distance_) {
        statistics->min_distance_ = min_distance;
    }
}

#endif