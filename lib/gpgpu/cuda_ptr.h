#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <utility>

namespace cpt {

template<typename T>
struct CudaDeleter
{
    void operator()(T* obj) {
        if (obj) {
            obj->~T();
        }
        cudaFree(obj);
    }
};

template<typename T,typename ...Args>
T* cuda_new(Args&&... args) {
    void* mem;
    cudaMallocManaged(&mem, sizeof(T));

    T* obj = new (mem) T(std::forward<Args>(args)...);
    return obj;
}

template<typename T,typename ...Args>
std::shared_ptr<T> cuda_make_shared(Args&&... args) {
    return std::shared_ptr<T>(cuda_new<T>(std::forward<Args>(args)...), CudaDeleter<T>());
}

}
