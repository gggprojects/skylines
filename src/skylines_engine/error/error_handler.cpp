
#include "error/error_handler.hpp"
#include "log/logger.hpp"

namespace sl { namespace error {
    ErrorHandler::ErrorHandler(std::string logger, std::string severity, std::shared_ptr<ThreadsErrors> thread_errors) :
        thread_errors_(thread_errors) {
        log::Logger::AddLogger(logger, severity);
    }
}}

