#pragma once

#include "gpgpu/math/cuda_vector.h"
#include "gpgpu/cuda_utils.h"

namespace cpt {

class CudaRay
{
public:
    CUDA_DEVHOST void set_origin(const CudaVector<float,3>& origin);
    CUDA_DEVHOST void set_direction(const CudaVector<float,3>& direction);
    CUDA_DEVHOST void set_max_t(float t) { _max_t = t; }

    CUDA_DEVHOST const CudaVector<float,3>& origin() const { return _origin; }
    CUDA_DEVHOST const CudaVector<float,3>& direction() const { return _direction; }
    CUDA_DEVHOST float max_t() const { return _max_t; }

    CUDA_DEVHOST bool is_alive() const { return _alive; }
    CUDA_DEVHOST void set_alive(bool b) { _alive = b; }

private:
    CudaVector<float,3> _origin;
    CudaVector<float,3> _direction;
    bool                _alive{true};

    float               _max_t{0.f};
};

}
