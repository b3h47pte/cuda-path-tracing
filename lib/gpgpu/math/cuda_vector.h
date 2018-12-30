#pragma once

#include <algorithm>
#include <Eigen/Core>
#include "gpgpu/cuda_utils.h"

namespace cpt {

template<typename T,int Dim>
class CudaVector
{
public:
    CudaVector() {
        for (auto i = 0; i < Dim; ++i) {
            _data[i] = T(0.0);
        }
    }

    void set_from_raw(const T* raw) {
        CHECK_CUBLAS_ERROR(cublasSetVector(Dim, sizeof(T), raw, 1, _data, 1));
    }

    CudaVector(const CudaVector& other)  {
        *this = other;
    }

    CudaVector& operator=(const CudaVector& other) {
        set_from_raw(other._data);
        return *this;
    }

    CudaVector(CudaVector&& other) {
        std::swap(_data, other._data);
    }

    CudaVector& operator=(CudaVector&& other) {
        std::swap(_data, other._data);
        return *this;
    }

private:
    T _data[Dim];
};

template<typename T,int Dim>
CudaVector<T,Dim> eigen_vector_to_cuda(const Eigen::Ref<const Eigen::Matrix<T,Dim,1>>& vec) {
    CudaVector<T,Dim> ret;
    ret.set_from_raw(vec.data());
    return ret;
}

}
