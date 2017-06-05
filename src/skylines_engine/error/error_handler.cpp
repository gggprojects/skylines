
#include "error/error_handler.hpp"

namespace skylines { namespace error {
    ErrorHandler::ErrorHandler() { 
        i = 0;
    }

    int ErrorHandler::GetValue() {
        return i;
    }

}}

