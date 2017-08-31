
#ifndef SKYLINES_GPU_GPU_MEMEMORY_HPP
#define SKYLINES_GPU_GPU_MEMEMORY_HPP

#include <vector>

#pragma warning(push, 0)
#include <cuda_runtime.h>
#pragma warning(pop)

#include "gpu/gpu_stream.hpp"

namespace sl { namespace gpu {

    template<class T>
    class GPUMemory {
    public:
        GPUMemory() : data_(nullptr) {
            num_elements_ = 0;
        }

        GPUMemory(size_t num_elements) : num_elements_(num_elements) {
            Malloc(num_elements_);
        }

        GPUMemory(const std::vector<T> &data, const GPUStream &gpu_stream) :
            GPUMemory(data.size()) {
            UploadToDeviceAsync(data.data(), gpu_stream);
        }

        ~GPUMemory() {
            if (data_ != nullptr) {
                cudaError_t e = cudaFree(data_);
            }
        }


        T * operator()() const { return data_; }

        void Malloc(size_t num_elements) {
            cudaError_t e = cudaMalloc((void**)&data_, num_elements * sizeof(T));
            // check error
        }

        void UploadToDeviceAsync(const std::vector<T> &source, const GPUStream &gpu_stream) {
            UploadToDeviceAsync(source.data(), source.size(), gpu_stream);
        }

        void UploadToDeviceAsync(const T *source, size_t num_elements, const GPUStream &gpu_stream) {
            MemCopyAsync(data_, source, num_elements, cudaMemcpyKind::cudaMemcpyHostToDevice, gpu_stream);
        }

        void UploadToDeviceAsync(const T *source, const GPUStream &gpu_stream) {
            UploadToDeviceAsync(source, num_elements_, gpu_stream);
        }

        void DownloadToHostAsync(T *destination, size_t num_elements, const GPUStream &gpu_stream) {
            MemCopyAsync(destination, data_, num_elements, cudaMemcpyKind::cudaMemcpyDeviceToHost, gpu_stream);
        }

        void DownloadToHostAsync(T *destination, const GPUStream &cudaStream) {
            DownloadToHostAsync(destination, num_elements_, cudaStream);
        }

        void DownloadToHost(T *destination, size_t num_elements) {
            MemCopy(destination, data_, num_elements, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        }

        void DownloadToHost(T *destination) {
            DownloadToHost(destination, num_elements_);
        }

    private:

        void MemCopyAsync(T *destination, const T *source, size_t num_elements, cudaMemcpyKind kind, const GPUStream &gpu_stream) {
            cudaError_t e = cudaMemcpyAsync(destination, source, num_elements * sizeof(T), kind, gpu_stream());
            //check error
        }

        void MemCopy(T *destination, const T *source, size_t num_elements, cudaMemcpyKind kind) {
            cudaError_t e = cudaMemcpy(destination, source, num_elements * sizeof(T), kind);
            //check error
        }

        T *data_;
        size_t num_elements_;
    };
}}

#endif
