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
    CUDA_DEVHOST CudaVector() {
        for (auto i = 0; i < Dim; ++i) {
            _data[i] = T(0.0);
        }
    }

    CUDA_DEVHOST CudaVector(const T* data) {
        set_from_raw(data);
    }

    CUDA_DEVHOST void set_from_raw(const T* raw) {
        memcpy(_data, raw, sizeof(T) * Dim);
    }

    CUDA_DEVHOST CudaVector(const CudaVector& other)  {
        set_from_raw(other._data);
    }

    CUDA_DEVHOST CudaVector& operator=(const CudaVector& other) {
        set_from_raw(other._data);
        return *this;
    }

    CUDA_DEVHOST CudaVector(CudaVector&& other) {
        set_from_raw(other._data);
    }

    CUDA_DEVHOST CudaVector& operator=(CudaVector&& other) {
        set_from_raw(other._data);
        return *this;
    }

    CUDA_DEVHOST static CudaVector Ones() {
        CudaVector vec;
        for (auto i = 0; i < Dim; ++i) {
            vec[i] = T(1.0);
        }
        return vec;
    }

    CUDA_DEVHOST static CudaVector Constant(T value) {
        CudaVector vec;
        for (auto i = 0; i < Dim; ++i) {
            vec[i] = value;
        }
        return vec;
    }

    CUDA_DEVHOST const T& operator[](size_t idx) const {
        return _data[idx];
    }

    CUDA_DEVHOST T& operator[](size_t idx) {
        return _data[idx];
    }

    CUDA_DEVHOST float norm() const {
        float ret = 0.f;
        for (auto i = 0; i < Dim; ++i) {
            ret += _data[i] * _data[i];
        }
        return sqrt(ret);
    }

    CUDA_DEVHOST void normalize() {
        const float mag = norm();
        for (auto i = 0; i < Dim; ++i) {
            _data[i] /= mag;
        }
    }

    CUDA_DEVHOST CudaVector& operator+=(const CudaVector& other) {
        for (auto i = 0; i < Dim; ++i) {
            _data[i] += other[i];
        }
        return *this;
    }

    CUDA_DEVHOST CudaVector& operator-=(const CudaVector& other) {
        for (auto i = 0; i < Dim; ++i) {
            _data[i] -= other[i];
        }
        return *this;
    }

private:
    T _data[Dim];
};

template<typename T,int Dim>
CUDA_HOST CudaVector<T,Dim> eigen_vector_to_cuda(const Eigen::Ref<const Eigen::Matrix<T,Dim,1>>& vec) {
    CudaVector<T,Dim> ret(vec.data());
    return ret;
}

template<typename T,int Dim>
CUDA_DEVHOST CudaVector<T,Dim> min(const CudaVector<T,Dim>& a, const CudaVector<T,Dim>& b) {
    CudaVector<T,Dim> ret;
    for (int i = 0; i < Dim; ++i) {
        ret[i] = (a[i] < b[i]) ? a[i] : b[i];
    }
    return ret;
}

template<typename T,int Dim>
CUDA_DEVHOST CudaVector<T,Dim> max(const CudaVector<T,Dim>& a, const CudaVector<T,Dim>& b) {
    CudaVector<T,Dim> ret;
    for (int i = 0; i < Dim; ++i) {
        ret[i] = (a[i] > b[i]) ? a[i] : b[i];
    }
    return ret;
}

template<typename T,int Dim>
CUDA_DEVHOST CudaVector<T,Dim> operator+(const CudaVector<T,Dim>& a, const CudaVector<T,Dim>& b) {
    CudaVector<T,Dim> ret = a;
    ret += b;
    return ret;
}

template<typename T,int Dim>
CUDA_DEVHOST CudaVector<T,Dim> operator-(const CudaVector<T,Dim>& a) {
    CudaVector<T,Dim> ret;
    for (int i = 0; i < Dim; ++i) {
        ret[i] = -a[i];
    }
    return ret;
}

template<typename T,int Dim>
CUDA_DEVHOST CudaVector<T,Dim> operator-(const CudaVector<T,Dim>& a, const CudaVector<T,Dim>& b) {
    return a + (-b);
}

template<typename T,int Dim,typename Scalar>
CUDA_DEVHOST CudaVector<T,Dim> operator*(const CudaVector<T,Dim>& a, Scalar scalar) {
    CudaVector<T,Dim> ret;
    for (int i = 0; i < Dim; ++i) {
        ret[i] = a[i] * scalar;
    }
    return ret;
}

template<typename T,int Dim,typename Scalar>
CUDA_DEVHOST CudaVector<T,Dim> operator*(Scalar scalar, const CudaVector<T,Dim>& a) {
    return a * scalar;
}

template<typename T,int Dim,typename Scalar>
CUDA_DEVHOST CudaVector<T,Dim> operator/(const CudaVector<T,Dim>& a, Scalar scalar) {
    CudaVector<T,Dim> ret;
    for (int i = 0; i < Dim; ++i) {
        ret[i] = a[i] / scalar;
    }
    return ret;
}

template<typename T,int Dim,typename Scalar>
CUDA_DEVHOST CudaVector<T,Dim> operator/(Scalar scalar, const CudaVector<T,Dim>& a) {
    CudaVector<T,Dim> ret;
    for (int i = 0; i < Dim; ++i) {
        ret[i] = scalar / a[i];
    }
    return ret;
}

template<typename T,int Dim>
CUDA_DEVHOST float dot(const CudaVector<T,Dim>& a, const CudaVector<T,Dim>& b) {
    float sum = 0.f;
    for (auto i = 0; i < Dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

template<typename T>
CUDA_DEVHOST CudaVector<T,3> cross(const CudaVector<T,3>& a, const CudaVector<T,3>& b) {
    CudaVector<T,3> ret;
    ret[0] = a[1] * b[2] - a[2] * b[1];
    ret[1] = a[2] * b[0] - a[0] * b[2];
    ret[2] = a[0] * b[1] - a[1] * b[0];
    return ret;
}

}
