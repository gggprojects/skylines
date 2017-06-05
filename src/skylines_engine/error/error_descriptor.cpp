#include <string>
#include "error/error_descriptor.hpp"
#include "error/return_codes.hpp"

namespace sl { namespace error {
    std::string SkylinesError::ErrorMessage() const {
        return std::move(GetReturnCodeString(static_cast<ReturnCode>(code_)));
    }

    std::string OpenGLError::ErrorMessage() const {
        return "";
    }

    std::string CudaError::ErrorMessage() const {
        return "";
    }
}}

