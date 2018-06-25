#ifndef SKYLINES_QUERIES_DATA_STADISTICS_HPP
#define SKYLINES_QUERIES_DATA_STADISTICS_HPP

#include <cuda_runtime.h>
#include "queries/data/data_structures.hpp"

namespace sl { namespace queries { namespace data {
    struct __align__(16) Statistics {

        __host__ __device__ Statistics() :
            num_comparisions_(0), output_size_(0) {
        }

        size_t num_comparisions_;
        size_t output_size_;
    };
}}}

//__host__ __device__ inline void IncrementStatistics(sl::queries::data::PointStatistics *statistics) {
//    statistics->num_comparisions_++;
//}

//__host__ __device__ inline void UpdateStatistics(
//    float distance,
//    sl::queries::data::PointStatistics *statistics) {
//    //the max distance
//    if (distance > statistics->max_distance_) {
//        statistics->max_distance_ = distance;
//    }
//
//    //the min distance
//    if (distance < statistics->min_distance_) {
//        statistics->min_distance_ = distance;
//    }
//}

#endif