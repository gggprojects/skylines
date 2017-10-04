#ifndef SKYLINES_LOGGER_HPP
#define SKYLINES_LOGGER_HPP

#include "spdlog/logger.h"

namespace sl { namespace log {

    using Logger_ptr = std::shared_ptr<spdlog::logger>;
    class Logger {
    public:
        static Logger_ptr AddLogger(const std::string &logger_name, const std::string &severity);
        static void SetSeverity(Logger_ptr logger_ptr_, const std::string &severity);
    private:
        static std::shared_ptr<spdlog::sinks::stdout_sink_mt> log_sinker_;
    };
}}

#endif // !SKYLINES_LOGGER_HPP
