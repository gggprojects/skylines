#include "queries/input_data.hpp"
#include "queries/data/random_generator.hpp"

namespace sl { namespace queries {
    void InputData::Render() {
        glColor3f(1, 0, 0);
        glPointSize(3);
        glBegin(GL_POINTS);
        for (const data::Point &p : points_) {
            glVertex2f(p.x_, p.y_);
        }
        glEnd();
    }

    void InputData::InitRandom(size_t num_points) {
        data::UniformRealRandomGenerator r;
        points_.clear();
        points_.reserve(num_points);
        for (size_t i = 0; i < num_points; i++) {
            points_.emplace_back(data::Point(r.Next(), r.Next()));
        }
    }
}}
