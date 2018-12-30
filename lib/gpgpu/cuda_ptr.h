#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <utility>
#include "gpgpu/cuda_utils.h"

namespace cpt {

template<typename T,typename ...Args>
T* cuda_new(Args&&... args) {
    void* mem;
    CHECK_CUDA_ERROR(cudaMallocManaged(&mem, sizeof(T)));

    T* obj = new (mem) T(std::forward<Args>(args)...);
    return obj;
}

template<typename T>
void cuda_delete(T* obj) {
    if (obj) {
        obj->~T();
    }
    CHECK_CUDA_ERROR(cudaFree(obj));
}

template<typename T,typename ...Args>
std::shared_ptr<T> cuda_make_shared(Args&&... args) {
    return std::shared_ptr<T>(cuda_new<T>(std::forward<Args>(args)...), cuda_delete<T>);
}

}
