#include <string>

#pragma warning(push, 0)
#include <cuda_runtime.h>
#pragma warning(pop)

#include "error/error_descriptor.hpp"
#include "error/return_codes.hpp"

namespace sl { namespace error {
    std::string SkylinesError::ErrorMessage() const {
        return std::move(GetReturnCodeString(static_cast<ReturnCode>(code_)));
    }

    std::string OpenGLError::ErrorMessage() const {
        return "";// gluErrorString(code_);
    }

    std::string CudaError::ErrorMessage() const {
        return cudaGetErrorString(static_cast<cudaError_t>(code_));
    }
}}

