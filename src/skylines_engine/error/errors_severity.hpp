#ifndef SKYLINES_ERROR_HPP
#define SKYLINES_ERROR_HPP

namespace sl { namespace error {

    enum class ErrorSeverity {
        OK = 0, // everything is fine
        WARN = 1, // unusual event, but no further impact
        ERRORS = 2, // request cannot be processed, but system is not affected
        CRITICAL = 4, // system affected, service downgrade (e.g. read-only)
        FATAL = 8  // system must be stopped
    };
}}

#endif
