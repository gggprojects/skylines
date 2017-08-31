
#ifndef SKYLINES_IRENDERABLE_HPP
#define SKYLINES_IRENDERABLE_HPP

#pragma warning(push, 0)
#include <QOpenGLFunctions>
#pragma warning(pop)

#include "export_import.hpp"

namespace sl { namespace common { 
    class skylines_engine_DLL_EXPORTS IRenderable {
    public:
        virtual void Render() const = 0;
    };
}}

#endif // !SKYLINES_IRENDERABLE_HPP
