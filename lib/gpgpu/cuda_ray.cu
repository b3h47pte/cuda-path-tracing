#include "cuda_ray.h"

namespace cpt {

void CudaRay::set_origin(const CudaVector<float,3>& origin) {
    _origin = origin;
}

void CudaRay::set_direction(const CudaVector<float,3>& direction) {
    _direction = direction;
    _direction.normalize();
}

}
