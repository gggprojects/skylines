#ifndef SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP
#define SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP

#include "common/irenderable.hpp"
#include "queries/data/random_generator.hpp"

namespace sl { namespace queries {namespace data {
    struct Point : public common::IRenderable {

        //random initializer
        Point(data::UniformRealRandomGenerator &r) : Point(r.Next(), r.Next()) {
        }

        Point(float x, float y) : x_(x), y_(y) {
        }

        void Render() const final {
            glVertex2f(x_, y_);
        }
        float x_;
        float y_;
    };

    struct WeightedPoint : public common::IRenderable {

        //random initializer
        WeightedPoint(data::UniformRealRandomGenerator &r) : WeightedPoint(Point(r), r.Next()) {
        }

        WeightedPoint(const Point &point, float weight) :
            point_(point),
            weight_(weight) {
        }

        void Render() const final {
            glVertex2f(point_.x_, point_.y_);
            //render weight
        }

        Point point_;
        float weight_;
    };
}}}

#endif // !SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP
