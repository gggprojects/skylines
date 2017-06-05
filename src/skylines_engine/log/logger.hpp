#ifndef SKYLINES_LOGGER_HPP
#define SKYLINES_LOGGER_HPP

#include "spdlog/logger.h"

namespace sl { namespace log {
    class Logger {
    public:
        static void AddLogger(const std::string &logger_name, std::string severity);
    private:
        static std::shared_ptr<spdlog::sinks::stdout_sink_mt> log_sinker_;
    };
}}

#endif // !SKYLINES_LOGGER_HPP
