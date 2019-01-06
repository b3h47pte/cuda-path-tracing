#pragma once

#include <curand_kernel.h>
#include "gpgpu/cuda_utils.h"
#include "gpgpu/cuda_ptr.h"
#include "gpgpu/math/cuda_vector.h"

namespace cpt {

class CudaSampler
{
public:
    CUDA_DEVICE CudaSampler(unsigned long long seed = 0);
    CUDA_DEVICE ~CudaSampler();

    CUDA_DEVICE virtual float sample_1d(float min_value, float max_value) = 0;
    CUDA_DEVICE virtual void sample_2d(
        CudaVector<float,2>& output,
        const CudaVector<float,2>& min_value,
        const CudaVector<float,2>& max_value) = 0;

protected:
    curandState_t _curand_state;

private:
    CUDA_DEVICE void generate_curand_state(unsigned long long seed);
    size_t _total;
};

}
