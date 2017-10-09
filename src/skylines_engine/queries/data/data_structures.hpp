#ifndef SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP
#define SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP

#include <cuda_runtime.h>

#include "common/irenderable.hpp"
#include "queries/data/random_generator.hpp"

namespace sl { namespace queries { namespace data {

    struct __align__(16) Statistics {

        Statistics() : num_comparisions_(0), output_size_(0) {
        }

        Statistics operator+=(const Statistics &other) {
            num_comparisions_ += other.num_comparisions_;
            output_size_ += other.output_size_;
            return *this;
        }

        size_t num_comparisions_;
        size_t output_size_;
    };

    struct __align__(8) Point {

        Point() {}

        //random initializer
        Point(data::UniformRealRandomGenerator &r) : Point(static_cast<float>(r.Next()), static_cast<float>(r.Next())) {
        }

        Point(float x, float y) : x_(x), y_(y) {
        }

        __device__ Point(const Point &other) : x_(other.x_), y_(other.y_) {
        }

        bool operator==(const Point &other) const {
            return x_ == other.x_ && y_ == other.y_;
        }

        __host__ __device__ inline float Distance(const Point &other) const {
            return std::sqrtf(SquaredDistance(other));
        }

        __host__ __device__ inline float SquaredDistance(const Point &other) const {
            return (x_ - other.x_) * (x_ - other.x_) + (y_ - other.y_) * (y_ - other.y_);
            //fast-math in CUDA computes __powf(x,y) as __exp2f(y * __log2f(x))
            //the log2 of a negative number causes an error.
            //return std::powf(x_ - other.x_, 2) + std::powf(y_ - other.y_, 2);
        }

        float x_;
        float y_;
    };

    struct __align__(16) WeightedPoint {

        #define MAX_WEIGHT 10

        WeightedPoint() {}

        //WeightedPoint(data::UniformRealRandomGenerator &r) : WeightedPoint(Point(r), static_cast<int>(r.Next() * 10) % 10) {
        WeightedPoint(data::UniformRealRandomGenerator &r) : WeightedPoint(Point(r), 1) {
        }

        WeightedPoint(const Point &point, int weight) :
            point_(point),
            weight_(weight) {
        }

        __host__ __device__ WeightedPoint(const WeightedPoint &other) :
            point_(other.point_),
            weight_(other.weight_) {
        }

        template<class Comparator>
        bool IsDominated(const WeightedPoint &other, const std::vector<Point> &q, Comparator comparator, Statistics *statistics) const {
            for (const Point p_q : q) {
                float distance = Distance(p_q);
                float other_distance = other.Distance(p_q);
                if (comparator(distance, other_distance)) {
                    return false;
                }
            }
            return true;
        }

        bool operator==(const WeightedPoint &other) const {
            return point_ == other.point_ && weight_ == other.weight_;
        }

        bool operator!=(const WeightedPoint &other) const {
            return !(*this == other);
        }

        __host__ __device__ inline float Distance(const Point &other) const {
            return point_.Distance(other) * weight_;
        }

        __host__ __device__ inline float SquaredDistance(const Point &other) const {
            return point_.SquaredDistance(other) * (weight_ * weight_);
        }


        Point point_;
        int weight_;
    };
}}}

#endif // !SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP
