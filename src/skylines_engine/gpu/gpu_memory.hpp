
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

        GPUMemory(size_t num_elements) {
            Malloc(num_elements);
        }

        ~GPUMemory() {
            if (data_ != nullptr) {
                cudaError_t e = cudaFree(data_);
                if (e != cudaSuccess) {
                    std::cout << cudaGetErrorString(e) << '\n';
                }
            }
            num_elements_ = 0;
        }

        T * operator()() const { return data_; }

        void Malloc(size_t num_elements) {
            num_elements_ = num_elements;
            cudaError_t e = cudaMalloc((void**)&data_, num_elements_ * sizeof(T));
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << '\n';
            }
        }

        //upload sync (had to be previous malloced)
        void UploadToDevice(const std::vector<T> &source) {
            UploadToDevice(source.data(), source.size());
        }
        void UploadToDevice(const T *source, size_t num_elements) {
            MemCopy(data_, source, num_elements, cudaMemcpyKind::cudaMemcpyHostToDevice);
        }
        void UploadToDevicePrimitive(const void *source, size_t num_elements) {
            cudaError_t e = cudaMemcpy(data_, source, num_elements, cudaMemcpyKind::cudaMemcpyHostToDevice);
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << '\n';
            }
        }

        //upload async (had to be previous malloced)
        void UploadToDeviceAsync(const std::vector<T> &source, const GPUStream &gpu_stream) {
            UploadToDeviceAsync(source.data(), source.size(), gpu_stream);
        }
        void UploadToDeviceAsync(const T *source, size_t num_elements, const GPUStream &gpu_stream) {
            MemCopyAsync(data_, source, num_elements, cudaMemcpyKind::cudaMemcpyHostToDevice, gpu_stream);
        }
        void UploadToDeviceAsync(const std::vector<T> &source, size_t offset, const GPUStream &gpu_stream) {
            MemCopyAsync(&data_[offset], source.data(), source.size(), cudaMemcpyKind::cudaMemcpyHostToDevice, gpu_stream);
        }


        //download sync (had to be previous malloced)
        void DownloadToHost(T *destination) {
            DownloadToHost(destination, num_elements_);
        }
        void DownloadToHost(T *destination, size_t num_elements) {
            MemCopy(destination, data_, num_elements, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        }
        void DownloadToHostAsync(T *destination, const GPUStream &cudaStream) {
            DownloadToHostAsync(destination, num_elements_, cudaStream);
        }
        void DownloadToHostAsync(T *destination, size_t num_elements, const GPUStream &gpu_stream) {
            MemCopyAsync(destination, data_, num_elements, cudaMemcpyKind::cudaMemcpyDeviceToHost, gpu_stream);
        }

    private:
        void MemCopyAsync(T *destination, const T *source, size_t num_elements, cudaMemcpyKind kind, const GPUStream &gpu_stream) {
            cudaError_t e = cudaMemcpyAsync(destination, source, num_elements * sizeof(T), kind, gpu_stream());
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << '\n';
            }
        }

        void MemCopy(T *destination, const T *source, size_t num_elements, cudaMemcpyKind kind) {
            cudaError_t e = cudaMemcpy(destination, source, num_elements * sizeof(T), kind);
            if (e != cudaSuccess) {
                std::cout << cudaGetErrorString(e) << '\n';
            }
        }

        T *data_;
        size_t num_elements_;
    };
}}

#endif
