#include "cuda_affine_transform.h"

namespace cpt {

CudaAffineTransform::CudaAffineTransform() {
}

CudaAffineTransform::CudaAffineTransform(const CudaMatrix<float,3,3>& linear, const CudaVector<float,3>& trans):
    _linear(linear),
    _trans(trans) {
}

CudaAffineTransform::CudaAffineTransform(const Transform& xform):
    CudaAffineTransform(
        eigen_matrix_to_cuda<float,3,3>(xform.rotation() * xform.scale().asDiagonal()),
        eigen_vector_to_cuda<float,3>(xform.translation())) {
}

}
