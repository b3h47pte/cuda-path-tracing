#pragma once

#include "gpgpu/cuda_aov_output.h"
#include "gpgpu/cuda_ray.h"
#include "gpgpu/cuda_scene.h"
#include "gpgpu/cuda_utils.h"

namespace cpt {

class CudaPathIntegrator
{
public:
    void Li(CudaRay* rays, size_t num_rays, const CudaScene* scene, size_t sampleIdx, CudaAovOutput* output) const;
};

}
