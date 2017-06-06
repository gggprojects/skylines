#ifndef SKYLINES_ERROR_HANDLER_HPP
#define SKYLINES_ERROR_HANDLER_HPP

#include <string>
#include <memory>
#include <string.h>

#include "export_import.hpp"
#include "error/threads_errors.hpp"
#include "error/error_descriptor.hpp"
#include "log/logger.hpp"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    #define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
    #define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define SL_LOG_DEBUG(x)  LogDebug(x, __FILENAME__, __LINE__)
#define SL_LOG_INFO(x)   LogInfo(x, __FILENAME__, __LINE__)
#define SL_LOG_WARN(x)   LogWarn(x, __FILENAME__, __LINE__)
#define SL_LOG_ERROR(x)  LogError(x, __FILENAME__, __LINE__)

namespace sl { namespace error {

    class skylines_engine_DLL_EXPORTS ErrorHandler {
    public:
        ErrorHandler(const std::string &logger, const std::string &severity, ThreadErrors_ptr thread_errors);

        void PushError(ErrorDescriptor_ptr err) {
            thread_errors_->PushError(err);
        }

        std::vector<std::string> GetErrors() {
            return std::move(thread_errors_->GetErrors().GetErrors());
        }

        template<class T>
        inline void LogDebug(T &argc, const char *file, const int line) {
            logger_ptr_->debug("{0}:{1} {2}", file, line, argc);
        }

        template<class T>
        inline void LogInfo(T &argc, const char *file, const int line) {
            logger_ptr_->info("{0}:{1} {2}", file, line, argc);
        }

        template<class T>
        inline void LogWarn(T &argc, const char *file, const int line) {
            logger_ptr_->warn("{0}:{1} {2}", file, line, argc);
        }

        template<class T>
        inline void LogError(T &argc, const char *file, const int line) {
            logger_ptr_->error("{0}:{1} {2}", file, line, argc);
        }

        void SetSeverity(std::string severity) {
            log::Logger::SetSeverity(logger_ptr_, severity);
        }
    private:
        ThreadErrors_ptr thread_errors_;
        log::Logger_ptr logger_ptr_;
    };
}}

#endif // SKYLINES_ERROR_HANDLER