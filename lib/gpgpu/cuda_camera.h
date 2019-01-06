#pragma once

#include <cstddef>
#include "gpgpu/cuda_object.h"
#include "gpgpu/cuda_ray.h"
#include "gpgpu/cuda_sampler.h"

namespace cpt {

class CudaCamera: public CudaObject
{
public:
    virtual void generate_rays(CudaSampler* samplers, CudaRay* rays, size_t width, size_t height) const = 0;
};

}
