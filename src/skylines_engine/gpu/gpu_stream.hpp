
#ifndef SKYLINES_GPU_GPU_STREAM_HPP
#define SKYLINES_GPU_GPU_STREAM_HPP

#include <cuda_runtime.h>

namespace sl { namespace gpu {

    class GPUStream {
    public:
        GPUStream() : stream_(nullptr) {
            cudaError_t e = cudaStreamCreate(&stream_);
        }
        ~GPUStream() {
            if (stream_ != nullptr) {
                cudaError_t e = cudaStreamDestroy(stream_);
                stream_ = nullptr;
            }
        }

        cudaStream_t GetStream() const { return stream_; }
    private:
        cudaStream_t stream_;
    };
}}

#endif
