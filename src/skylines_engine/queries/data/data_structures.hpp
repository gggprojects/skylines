#ifndef SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP
#define SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP

#include <cuda_runtime.h>

#include "common/irenderable.hpp"
#include "queries/data/random_generator.hpp"
#include "queries/data/stadistics.hpp"

namespace sl { namespace queries { namespace data {

    struct Range {
        double min_x;
        double max_x;
        double min_y;
        double max_y;
    };

    struct __align__(8) Point {

        Point() {}

        Point(
            data::UniformRealRandomGenerator &rrg_x,
            data::UniformRealRandomGenerator &rrg_y) :
            Point(static_cast<float>(rrg_x.Next()), static_cast<float>(rrg_y.Next())) {
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

        WeightedPoint(
            data::UniformRealRandomGenerator &rrg_x,
            data::UniformRealRandomGenerator &rrg_y,
            data::UniformIntRandomGenerator &irg) : WeightedPoint(Point(rrg_x, rrg_y), irg.Next() % 10) {
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
                float a_distance = Distance(p_q);
                float b_distance = other.Distance(p_q);
                ComputeStatistics(a_distance, b_distance, statistics);
                if (comparator(a_distance, b_distance)) {
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
