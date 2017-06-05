#ifndef SKYLINES_ERROR_HANDLER_HPP
#define SKYLINES_ERROR_HANDLER_HPP

#include <string>
#include <memory>

#include "export_import.hpp"
#include "error/threads_errors.hpp"
#include "error/error_descriptor.hpp"

namespace sl { namespace error {
    class skylines_engine_DLL_EXPORTS ErrorHandler {
    public:
        ErrorHandler(std::string logger, std::string severity, std::shared_ptr<ThreadsErrors> thread_errors);

        void PushError(std::shared_ptr<ErrorDescriptor> err) {
            thread_errors_->PushError(err);
        }

        std::vector<std::string> GetErrors() {
            return std::move(thread_errors_->GetErrors().GetErrors());
        }

    private:
        std::shared_ptr<ThreadsErrors> thread_errors_;
    };
}}

#endif // SKYLINES_ERROR_HANDLER