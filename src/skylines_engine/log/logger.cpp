
#include <string>
#include <memory>
#include <locale>
#include <iostream>

#include "spdlog/spdlog.h"

#include "log/logger.hpp"

namespace sl { namespace log {

    std::shared_ptr<spdlog::sinks::stdout_sink_mt> Logger::log_sinker_ = nullptr;

    std::string ToUpper(const std::string &str) {
        std::string ret(str);
        std::locale loc;
        for (std::string::size_type i = 0; i<str.length(); ++i)
            ret[i] = std::toupper(str[i], loc);
        return std::move(ret);
    }

    int GetSpdlogSeverityFromString(const std::string &str) {
        std::string str_upper = ToUpper(str);
        if (str_upper == "TRACE") {
            return spdlog::level::level_enum::trace;
        } else if (str_upper == "DEBUG") {
            return spdlog::level::level_enum::debug;
        } else if (str_upper == "INFO") {
            return spdlog::level::level_enum::info;
        } else if (str_upper == "WARN") {
            return spdlog::level::level_enum::warn;
        } else if (str_upper == "ERROR") {
            return spdlog::level::level_enum::err;
        } else if (str_upper == "CRITICAL") {
            return spdlog::level::level_enum::critical;
        } else {
            return -1;
        }
    }

    void Logger::AddLogger(const std::string &logger_name, std::string severity) {
        int severity_level_int = GetSpdlogSeverityFromString(severity);
        if (severity_level_int == -1) {
            std::cerr << "Unknow severity level " << severity << std::endl;
            return;
        }

        spdlog::level::level_enum spd_severity = static_cast<spdlog::level::level_enum>(severity_level_int);

        std::shared_ptr<spdlog::logger> ptr = spdlog::get(logger_name);
        if (ptr == nullptr) {
            const char* LOG_FORMAT = "[%D %H:%M:%S.%e %z] [%l] [thread %t] %v";
            ptr = std::make_shared<spdlog::logger>(logger_name, Logger::log_sinker_);
            spdlog::register_logger(ptr);
            ptr->set_pattern(LOG_FORMAT);
        }
        ptr->set_level(spd_severity);
    }
}}
