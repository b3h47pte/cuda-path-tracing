#pragma once

#include "scene/object.h"
#include "gpgpu/cuda_utils.h"
#include "gpgpu/math/cuda_vector.h"

namespace cpt {

class CudaObject
{
public:
    // world space
    CUDA_DEVHOST const CudaVector<float,3>& position() const { return _position; }

    CUDA_DEVHOST const CudaVector<float,3>& world_up_dir() const { return _world_up_dir; }
    CUDA_DEVHOST const CudaVector<float,3>& world_right_dir() const { return _world_right_dir; }
    CUDA_DEVHOST const CudaVector<float,3>& world_forward_dir() const { return _world_forward_dir; }

    void bake_from_object(const Object& object);

private:
    CudaVector<float,3> _position;

    CudaVector<float,3> _world_up_dir;
    CudaVector<float,3> _world_right_dir;
    CudaVector<float,3> _world_forward_dir;
};

}
