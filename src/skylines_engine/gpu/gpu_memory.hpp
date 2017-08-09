
#ifndef SKYLINES_GPU_GPU_MEMEMORY_HPP
#define SKYLINES_GPU_GPU_MEMEMORY_HPP

#include <cuda_runtime.h>

namespace sl { namespace gpu {
    template<class T>
    class GPUMemory {
    public:
        GPUMemory() : data_(nullptr) {
        }

        GPUMemory(size_t num_elements) {
            Malloc(num_elements);
        }

        ~GPUMemory() {
            if (data_ != nullptr) {
                cudaError_t e = cudaFree(data_);
            }
        }

        void Malloc(size_t num_elements) {
            cudaError_t e = cudaMalloc((void**)&data_, num_elements * sizeof(T));
        }

    private:
        T *data_;
    };
}}

#endif
