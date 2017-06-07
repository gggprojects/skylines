#ifndef SLYLINES_QUERIES_OUTPUT_DATA_HPP
#define SLYLINES_QUERIES_OUTPUT_DATA_HPP

#include "common/irenderable.hpp"

namespace sl { namespace queries {
    class OutputData : public common::IRenderable {
    public:
        void Render() final {
            glColor3f(0, 1, 0);
            glBegin(GL_POLYGON); {
                glVertex2f(0.25, 0.25);
                glVertex2f(0.75, 0.25);
                glVertex2f(0.5, 0.75);
            }glEnd();
        }
    private:

    };
}}

#endif // !SLYLINES_QUERIES_DATA_OUTPUT_DATA_HPP