#pragma once

#include "gpgpu/cuda_utils.h"
#include "gpgpu/math/cuda_vector.h"

namespace cpt {

template<typename T,int Rows,int Columns>
class CudaMatrix
{
public:
    CUDA_DEVHOST static CudaMatrix Identity() {
        CudaMatrix matrix;
        const int dim = (Rows > Columns) ? Columns : Rows;
        for (int r = 0; r < dim; ++r) {
            matrix(r, r) = 1.f;
        }
        return matrix;
    }

    CUDA_DEVHOST const CudaVector<T, Columns>& operator[](size_t r) const {
        return _data[r];
    }

    CUDA_DEVHOST CudaVector<T, Columns>& operator[](size_t r) {
        return _data[r];
    }

    CUDA_DEVHOST T& operator()(size_t r, size_t c) {
        return _data[r][c];
    }

    CUDA_DEVHOST const T& operator()(size_t r, size_t c) const {
        return _data[r][c];
    }

private:
    CudaVector<T, Columns> _data[Rows];
};

template<typename T,int Rows,int Columns>
CUDA_HOST CudaMatrix<T,Rows,Columns> eigen_matrix_to_cuda(const Eigen::Ref<const Eigen::Matrix<T,Rows,Columns>>& mat) {
    CudaMatrix<T,Rows,Columns> ret;
    for (auto r = 0; r < Rows; ++r) {
        for (auto c = 0; c < Columns; ++c) {
            ret(r, c) = mat(r, c);
        }
    }
    return ret;
}

template<typename T,int Rows,int Columns>
CUDA_DEVHOST CudaVector<T,Rows> operator*(const CudaMatrix<T,Rows,Columns>& A, const CudaVector<T,Columns>& x) {
    CudaVector<T,Rows> b;
    for (int i = 0; i < Rows; ++i) {
        b[i] = dot(A[i], x);
    }
    return b;
}



}
