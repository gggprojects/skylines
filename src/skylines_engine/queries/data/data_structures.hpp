#ifndef SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP
#define SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP

#include <cuda_runtime.h>

#include "common/irenderable.hpp"
#include "queries/data/random_generator.hpp"

namespace sl { namespace queries { namespace data {

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
            //return std::powf(x_ - other.x_, 2) + std::powf(y_ - other.y_, 2);
        }

        float x_;
        float y_;
    };

    struct __align__(16) WeightedPoint {

        WeightedPoint() {}

        WeightedPoint(data::UniformRealRandomGenerator &r) : WeightedPoint(Point(r), static_cast<float>(r.Next())) {
        //WeightedPoint(data::UniformRealRandomGenerator &r) : WeightedPoint(Point(r), 1.) {
        }

        WeightedPoint(const Point &point, float weight) :
            point_(point),
            weight_(weight) {
        }

        __host__ __device__ WeightedPoint(const WeightedPoint &other) :
            point_(other.point_),
            weight_(other.weight_) {
        }

        bool IsDominated(const WeightedPoint &other, const std::vector<Point> &q) const {
            for (const Point p_q : q) {
                float distance = Distance(p_q);
                float other_distance = other.Distance(p_q);
                if (distance <= other_distance) {
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
            return point_.Distance(other) / weight_;
        }

        __host__ __device__ inline float SquaredDistance(const Point &other) const {
            return point_.SquaredDistance(other) / (weight_ * weight_);
        }


        Point point_;
        float weight_;
    };
}}}

#endif // !SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP
