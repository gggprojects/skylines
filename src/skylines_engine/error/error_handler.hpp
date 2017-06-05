#ifndef SKYLINES_ERROR_HANDLER_HPP
#define SKYLINES_ERROR_HANDLER_HPP

#include "export_import.hpp"

namespace skylines { namespace error {
    class skylines_engine_DLL_EXPORTS ErrorHandler {
    public:
        ErrorHandler();

        int GetValue();
    private:
        int i;
    };
}}

#endif // SKYLINES_ERROR_HANDLER