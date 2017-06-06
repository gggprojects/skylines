#ifndef SKYLINE_SKYLINE_ELEMENT_HPP
#define SKYLINE_SKYLINE_ELEMENT_HPP

#include "error/error_handler.hpp"

namespace sl { namespace common {

    class skylines_engine_DLL_EXPORTS SkylineElement : public error::ErrorHandler {
    public:
        SkylineElement(const std::string &logger, const std::string &severity, error::ThreadErrors_ptr thread_errors) :
            ErrorHandler(logger, severity, thread_errors) {
        }
    };
}}

#endif // !SKYLINE_SKYLINE_ELEMENT_HPP
