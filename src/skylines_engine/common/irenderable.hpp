
#ifndef SKYLINES_IRENDERABLE_HPP
#define SKYLINES_IRENDERABLE_HPP

#include <QOpenGLFunctions>

namespace sl { namespace common { 
    class IRenderable {
    public:
        virtual void Render() = 0;
    };
}}

#endif // !SKYLINES_IRENDERABLE_HPP
