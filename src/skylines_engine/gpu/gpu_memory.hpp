
#ifndef SKYLINES_GPU_GPU_MEMEMORY_HPP
#define SKYLINES_GPU_GPU_MEMEMORY_HPP

#include <vector>

#include <cuda_runtime.h>

#include "gpu/gpu_stream.hpp"

namespace sl { namespace gpu {

    template<class T>
    class GPUMemory {
    public:
        GPUMemory() : data_(nullptr) {
        }

        GPUMemory(size_t num_elements) : num_elements_(num_elements) {
            Malloc(num_elements_);
        }

        GPUMemory(const std::vector<T> &data, const GPUStream &gpu_stream) {
            size_t num_elements = data.size();
            Malloc(num_elements);
            HostToDeviceAsync(data.data(), num_elements, gpu_stream);
        }

        ~GPUMemory() {
            if (data_ != nullptr) {
                cudaError_t e = cudaFree(data_);
            }
        }


        T * GetData() const { return data_; }

        void Malloc(size_t num_elements) {
            cudaError_t e = cudaMalloc((void**)&data_, num_elements * sizeof(T));
            // check error
        }

        void HostToDeviceAsync(const T *source, size_t num_elements, const GPUStream &gpu_stream) {
            MemCopyAsync(source, num_elements, cudaMemcpyKind::cudaMemcpyHostToDevice, gpu_stream);
        }

        void HostToDevideAsync(const T *source, const GPUStream &gpu_stream) {
            HostToDeviceAsync(source, num_elements_, gpu_stream);
        }

        void DeviceToHostAsync(const T *source, size_t num_elements, const GPUStream &gpu_stream) {
            MemCopyAsync(source, num_elements, cudaMemcpyKind::cudaMemcpyDeviceToHost, gpu_stream);
        }

        void DeviceToHostAsync(const T *source, const GPUStream &cudaStream) {
            DeviceToHostAsync(source, num_elements_, cudaStream);
        }

    private:

        void MemCopyAsync(const T *source, size_t num_elements, cudaMemcpyKind kind, const GPUStream &gpu_stream) {
            cudaError_t e = cudaMemcpyAsync(data_, source, num_elements * sizeof(T), kind, gpu_stream.GetStream());
            //check error
        }

        T *data_;
        size_t num_elements_;
    };
}}

#endif
