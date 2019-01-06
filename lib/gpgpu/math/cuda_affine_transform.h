#pragma once

#include "gpgpu/math/cuda_matrix.h"
#include "gpgpu/math/cuda_vector.h"
#include "math/transform.h"

namespace cpt {

class CudaAffineTransform
{
public:
    CudaAffineTransform();
    CudaAffineTransform(const CudaMatrix<float,3,3>& linear, const CudaVector<float,3>& trans);
    CudaAffineTransform(const Transform& xform);

    CUDA_DEVHOST CudaVector<float,3> transform(const CudaVector<float,3>& input, bool as_point) const;

private:
    CudaMatrix<float,3,3> _linear;
    CudaVector<float,3> _trans;
};

}
