#pragma once

#include "gpgpu/cuda_sampler.h"

namespace cpt {

class CudaUniformSampler: public CudaSampler
{
public:
    CUDA_DEVICE float sample_1d(float min_value, float max_value) override;
    CUDA_DEVICE void sample_2d(
        CudaVector<float,2>& output,
        const CudaVector<float,2>& min_value,
        const CudaVector<float,2>& max_value) override;
};

}
