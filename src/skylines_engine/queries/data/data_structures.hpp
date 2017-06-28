#ifndef SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP
#define SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP

#include "common/irenderable.hpp"
#include "queries/data/random_generator.hpp"

namespace sl { namespace queries {namespace data {

    struct Point;

    template<class T>
    struct Dominable {
        virtual float Distance(const Point &other) const = 0;
        virtual float SquaredDistance(const Point &other) const = 0;

        bool IsDominated(const T &other, const std::vector<Point> &q) const {
            for (const Point p_q : q) {
                float distance = this->Distance(p_q);
                float other_distance = other.Distance(p_q);
                if (distance <= other_distance) {
                    return false;
                }
            }
            return true;
        }
    };

    struct Point : public common::IRenderable, Dominable<Point> {

        Point() {}

        //random initializer
        Point(data::UniformRealRandomGenerator &r) : Point(r.Next(), r.Next()) {
        }

        Point(float x, float y) : x_(x), y_(y) {
        }

        void Render() const final {
            glVertex2f(x_, y_);
        }

        float Distance(const Point &other) const final {
            return std::sqrtf(SquaredDistance(other));
        }

        float SquaredDistance(const Point &other) const final {
            return std::powf(x_ - other.x_, 2) + std::powf(y_ - other.y_, 2);
        }

        float x_;
        float y_;
    };

    struct WeightedPoint : public common::IRenderable, Dominable<WeightedPoint> {

        WeightedPoint() {}

        WeightedPoint(data::UniformRealRandomGenerator &r) : WeightedPoint(Point(r), 1) {
        }

        WeightedPoint(const Point &point, float weight) :
            point_(point),
            weight_(weight) {
        }

        WeightedPoint(const WeightedPoint &other) :
            point_(other.point_),
            weight_(other.weight_) {
        }

        void Render() const final {
            glVertex2f(point_.x_, point_.y_);
            //render weight text
        }

        float Distance(const Point &other) const final {
            return point_.Distance(other) * weight_;
        }

        float SquaredDistance(const Point &other) const final {
            return point_.SquaredDistance(other) * weight_;
        }

        Point point_;
        float weight_;
    };
}}}

#endif // !SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP
