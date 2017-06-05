#ifndef SKYLINES_RETURN_CODES_HPP
#define SKYLINES_RETURN_CODES_HPP

#include <string>

#include "error/return_codes.hpp"

namespace sl { namespace error {

    enum class ReturnCode {
        OK = 0,
        RUNTIME_ERROR
    };

    inline static const std::string GetReturnCodeString(ReturnCode err_code) {
        switch (err_code) {
            case sl::error::ReturnCode::OK: return "OK";
            case sl::error::ReturnCode::RUNTIME_ERROR: return "RUNTIME_ERROR";
            default: return "Unknow error code";
        }
    }
}}

#endif
