
#ifndef SKYLINES_GPU_GPU_STREAM_HPP
#define SKYLINES_GPU_GPU_STREAM_HPP

#pragma warning(push, 0)
#include <cuda_runtime.h>
#pragma warning(pop)

namespace sl { namespace gpu {

    class GPUStream {
    public:
        GPUStream() : stream_(nullptr) {
            cudaError_t e = cudaStreamCreate(&stream_);
            //check error
        }
        ~GPUStream() {
            if (stream_ != nullptr) {
                cudaError_t e = cudaStreamDestroy(stream_);
                //check error
                stream_ = nullptr;
            }
        }

        cudaStream_t operator()() const { return stream_; }

        cudaError_t Syncronize() {
            return cudaStreamSynchronize(stream_);
        }

    private:
        cudaStream_t stream_;
    };
}}

#endif
