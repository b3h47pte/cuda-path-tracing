#pragma once

#include <algorithm>
#include <cuda_runtime.h>
#include <Eigen/Core>
#include "gpgpu/cuda_utils.h"
#include "gpgpu/cuda_ptr.h"
#include <iostream>

namespace cpt {

template<typename T,int Dim>
class CudaVector
{
public:
    CudaVector() {
        alloc();
        for (auto i = 0; i < Dim; ++i) {
            _data[i] = T(0.0);
        }
    }

    CudaVector(const T* data) {
        alloc();
        set_from_raw(data);
    }

    ~CudaVector() {
        cudaFree(_data);
    }

    void set_from_raw(const T* raw) {
        memcpy(_data, raw, sizeof(T) * Dim);
    }

    CudaVector(const CudaVector& other)  {
        alloc();
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

    const T& operator[](size_t idx) const {
        return _data[idx];
    }

    T& operator[](size_t idx) {
        return _data[idx];
    }

private:
    void alloc() {
        CHECK_CUDA_ERROR(cudaMallocManaged(&_data, sizeof(T) * Dim));
    }

    T* _data{nullptr};
};

template<typename T,int Dim>
CudaVector<T,Dim>* eigen_vector_to_cuda(const Eigen::Ref<const Eigen::Matrix<T,Dim,1>>& vec) {
    CudaVector<T,Dim>* ret = cuda_new<CudaVector<T,Dim>>(vec.data());
    return ret;
}

template<typename T,int Dim>
CudaVector<T,Dim> min(const CudaVector<T,Dim>& a, const CudaVector<T,Dim>& b) {
    CudaVector<T,Dim> ret;
    for (int i = 0; i < Dim; ++i) {
        ret[i] = (a[i] < b[i]) ? a[i] : b[i];
    }
    return ret;
}

template<typename T,int Dim>
CudaVector<T,Dim> max(const CudaVector<T,Dim>& a, const CudaVector<T,Dim>& b) {
    CudaVector<T,Dim> ret;
    for (int i = 0; i < Dim; ++i) {
        ret[i] = (a[i] > b[i]) ? a[i] : b[i];
    }
    return ret;
}

template<typename T,int Dim>
CudaVector<T,Dim> operator+(const CudaVector<T,Dim>& a, const CudaVector<T,Dim>& b) {
    CudaVector<T,Dim> ret;
    for (int i = 0; i < Dim; ++i) {
        ret[i] = a[i] + b[i];
    }
    return ret;
}

template<typename T,int Dim>
CudaVector<T,Dim> operator-(const CudaVector<T,Dim>& a) {
    CudaVector<T,Dim> ret;
    for (int i = 0; i < Dim; ++i) {
        ret[i] = -a[i];
    }
    return ret;
}

template<typename T,int Dim>
CudaVector<T,Dim> operator-(const CudaVector<T,Dim>& a, const CudaVector<T,Dim>& b) {
    return a + (-b);
}

template<typename T,int Dim,typename Scalar>
CudaVector<T,Dim> operator*(const CudaVector<T,Dim>& a, Scalar scalar) {
    CudaVector<T,Dim> ret;
    for (int i = 0; i < Dim; ++i) {
        ret[i] = a[i] * scalar;
    }
    return ret;
}

template<typename T,int Dim,typename Scalar>
CudaVector<T,Dim> operator*(Scalar scalar, const CudaVector<T,Dim>& a) {
    return a * scalar;
}

template<typename T,int Dim,typename Scalar>
CudaVector<T,Dim> operator/(const CudaVector<T,Dim>& a, Scalar scalar) {
    CudaVector<T,Dim> ret;
    for (int i = 0; i < Dim; ++i) {
        ret[i] = a[i] / scalar;
    }
    return ret;
}

template<typename T,int Dim,typename Scalar>
CudaVector<T,Dim> operator/(Scalar scalar, const CudaVector<T,Dim>& a) {
    CudaVector<T,Dim> ret;
    for (int i = 0; i < Dim; ++i) {
        ret[i] = scalar / a[i];
    }
    return ret;
}

}
