#include "cuda_affine_transform.h"

namespace cpt {

CUDA_DEVHOST CudaVector<float,3> CudaAffineTransform::transform(const CudaVector<float,3>& input, bool as_point) const {
    CudaVector<float,3> ret = _linear * input;
    if (as_point) {
        ret += _trans;
    }
    return ret;
}

}
