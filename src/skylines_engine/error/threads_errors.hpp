#ifndef SKYLINES_THREADS_ERRORS_HPP
#define SKYLINES_THREADS_ERRORS_HPP

#include <unordered_map>
#include <mutex>
#include <thread>

#include "error/thread_errors_stack.hpp"

namespace sl { namespace error {
    class ThreadsErrors {
    public:
        ThreadsErrors() {}

        std::shared_ptr<ThreadsErrors> Instanciate() {
            return std::make_shared<ThreadsErrors>();
        }

        ThreadsErrors(const ThreadsErrors &other) = delete;
        ThreadsErrors(ThreadsErrors &&other) = delete;

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
        std::mutex map_mutex_;
        std::unordered_map<std::thread::id, ThreadErrorsStack> thread_errors_map_;
    };
}}

#endif // !SKYLINES_THREADS_ERRORS_HPP
