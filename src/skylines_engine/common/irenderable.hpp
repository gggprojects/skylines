
#ifndef SKYLINES_IRENDERABLE_HPP
#define SKYLINES_IRENDERABLE_HPP

#include <QOpenGLFunctions>

#include "export_import.hpp"

namespace sl { namespace common { 
    class skylines_engine_DLL_EXPORTS IRenderable {
    public:
        virtual void Render() = 0;
    };
}}

#endif // !SKYLINES_IRENDERABLE_HPP
