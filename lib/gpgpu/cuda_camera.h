#pragma once

#include <cstddef>
#include "gpgpu/cuda_ray.h"

namespace cpt {

class CudaCamera
{
public:
    virtual void generate_rays(CudaRay* rays, size_t width, size_t height) const = 0;
};

}
