#ifndef SKYLINES_ERROR_DESCRIPTOR_HPP
#define SKYLINES_ERROR_DESCRIPTOR_HPP

#include <string>
#include "error/errors_severity.hpp"

namespace sl { namespace error {
    class ErrorDescriptor {
    public:
        ErrorDescriptor(ErrorSeverity severity, int code, const std::string &file, int line) :
            severity_(severity), code_(code), file_(file), line_(line) {
        }

        virtual std::string ErrorMessage() const = 0;
    protected:
        ErrorSeverity severity_;
        int code_;
        std::string file_;
        int line_;
    };

    class SkylinesError : public ErrorDescriptor {
    public:
        SkylinesError(ErrorSeverity severity, int code, const std::string &file, int line) :
            ErrorDescriptor(severity, code, file, line) {
        }

        std::string ErrorMessage() const final;
    };

    class OpenGLError : public ErrorDescriptor {
    public:
        OpenGLError(ErrorSeverity severity, int code, const std::string &file, int line) :
            ErrorDescriptor(severity, code, file, line) {
        }

        std::string ErrorMessage() const final;
    };

    class CudaError : public ErrorDescriptor {
    public:
        CudaError(ErrorSeverity severity, int code, const std::string &file, int line) :
            ErrorDescriptor(severity, code, file, line) {
        }

        std::string ErrorMessage() const final;
    };
}}

#endif // !SKYLINES_ERROR_DESCRIPTOR_HPP
