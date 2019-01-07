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

    CUDA_DEVHOST int dbg_x_idx() const { return _x_idx; }
    CUDA_DEVHOST void dbg_set_x_idx(int idx) { _x_idx = idx; }

    CUDA_DEVHOST int dbg_y_idx() const { return _y_idx; }
    CUDA_DEVHOST void dbg_set_y_idx(int idx) { _y_idx = idx; }

    CUDA_DEVHOST int dbg_flat_idx() const { return _flat_idx; }
    CUDA_DEVHOST void dbg_set_flat_idx(int idx) { _flat_idx = idx; }

private:
    CudaVector<float,3> _origin;
    CudaVector<float,3> _direction;
    bool                _alive{true};

    float               _max_t{0.f};

    // Debug info.
    int                 _x_idx{0};
    int                 _y_idx{0};
    int                 _flat_idx{0};
};

}
