#pragma once

#include "gpgpu/math/cuda_vector.h"

namespace cpt {

class CudaAABB
{
public:
    const CudaVector<float,3>& centroid() const { return _centroid; }

private:
    CudaVector<float,3> _centroid;
};

}
