#ifndef SKYLINES_THREAD_ERRORS_STACK_HPP
#define SKYLINES_THREAD_ERRORS_STACK_HPP

#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>

#include "error/error_descriptor.hpp"

namespace sl { namespace error {
    struct ThreadErrorsStack {
    public:
        void PushError(std::shared_ptr<ErrorDescriptor> error) {
            stack_errors_.emplace_back(error);
        }

        std::vector<std::shared_ptr<ErrorDescriptor>> stack_errors_;

        std::vector<std::string> GetErrors() {
            std::vector<std::string> errors;
            std::transform(stack_errors_.begin(), stack_errors_.end(), std::back_inserter(errors), [](std::shared_ptr<ErrorDescriptor> err) -> std::string {
                return std::move(err->ErrorMessage());
            });
            return std::move(errors);
        }
    };
}}

#endif // !SKYLINES_THREAD_ERRORS_QUEUE_HPP
