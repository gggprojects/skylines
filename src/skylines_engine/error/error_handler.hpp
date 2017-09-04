#ifndef SKYLINES_ERROR_HANDLER_HPP
#define SKYLINES_ERROR_HANDLER_HPP

#include <string>
#include <memory>
#include <unordered_map>
#include <string.h>

#pragma warning(push, 0)
#include <cuda_runtime.h>
#pragma warning(pop)

#include "export_import.hpp"
#include "error/error_descriptor.hpp"
#include "error/thread_errors_stack.hpp"
#include "log/logger.hpp"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    #define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
    #define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define SL_LOG_DEBUG(x)     LogDebug(x, __FILENAME__, __LINE__)
#define SL_LOG_INFO(x)      LogInfo(x, __FILENAME__, __LINE__)
#define SL_LOG_WARN(x)      LogWarn(x, __FILENAME__, __LINE__)
#define SL_LOG_ERROR(x)     LogError(x, __FILENAME__, __LINE__)

#define SL_PUSH_ERROR(x)    PushError(std::make_shared<sl::error::SkylinesError>(sl::error::ErrorSeverity::ERRORS, x, __FILENAME__, __LINE__))
#define GL_PUSH_ERROR(x)    PushError(std::make_shared<sl::error::OpenGLError>(sl::error::ErrorSeverity::ERRORS, x, __FILENAME__, __LINE__))
#define CUDA_PUSH_ERROR(x)  PushError(std::make_shared<sl::error::CudaError>(sl::error::ErrorSeverity::ERRORS, x, __FILENAME__, __LINE__))
#define CUDA_CHECK(x)       CudaCheck(x, __FILENAME__, __LINE__)
#define CUDA_CHECK_LAST_ERROR CUDA_CHECK(cudaGetLastError())

namespace sl { namespace error {

    class skylines_engine_DLL_EXPORTS ErrorHandler {
    private:
        class ThreadsErrors {
        public:
            ThreadsErrors(const ThreadsErrors &other) = delete;
            ThreadsErrors(ThreadsErrors &&other) = delete;

            ThreadsErrors& operator=(const ThreadsErrors &other) = delete;
            ThreadsErrors& operator=(ThreadsErrors &&other) = delete;

            static ThreadsErrors& GetInstance() {
                static ThreadsErrors instance;  // Guaranteed to be destroyed.
                                                // Instantiated on first use.
                return instance;
            }

            void PushError(std::shared_ptr<ErrorDescriptor> err) {
                std::lock_guard<std::mutex> lock_(map_mutex_);
                thread_errors_map_[std::this_thread::get_id()].PushError(err);
            }

            ThreadErrorsStack GetErrors() {
                ThreadErrorsStack tes;
                std::lock_guard<std::mutex> lock_(map_mutex_);
                auto it = thread_errors_map_.find(std::this_thread::get_id());
                if (it != thread_errors_map_.end()) {
                    tes = std::move(it->second);
                    thread_errors_map_.erase(it);
                }
                return std::move(tes);
            }
        private:
            ThreadsErrors() {}

            std::mutex map_mutex_;
            std::unordered_map<std::thread::id, ThreadErrorsStack> thread_errors_map_;
        };

    public:
        ErrorHandler(const std::string &logger, const std::string &severity);

        void PushError(std::shared_ptr<ErrorDescriptor> err) {
            ThreadsErrors::GetInstance().PushError(err);
            LogError(err);
        }

        std::vector<std::string> GetErrors() {
            return std::move(ThreadsErrors::GetInstance().GetErrors().GetErrors());
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

        inline void LogError(std::shared_ptr<ErrorDescriptor> err) {
            logger_ptr_->error("{0}:{1} {2}", err->file_, err->line_, err->ErrorMessage());
        }

        void SetSeverity(std::string severity) {
            log::Logger::SetSeverity(logger_ptr_, severity);
        }

        void CudaCheck(cudaError_t result, char const *file, const int line) {
            if (result != cudaSuccess) {
                CUDA_PUSH_ERROR(result);
#ifdef _DEBUG
                assert(true);
#endif
            }
        }
    private:
        log::Logger_ptr logger_ptr_;
    };
}}

#endif // SKYLINES_ERROR_HANDLER