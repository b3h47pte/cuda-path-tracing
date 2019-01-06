#include "cuda_uniform_sampler.h"

namespace cpt {

CUDA_DEVICE float CudaUniformSampler::sample_1d(float min_value, float max_value) {
    const int idx = get_cuda_flat_thread_index();
    const float num = curand_uniform(&_curand_state);
    return num * (max_value - min_value) + min_value;
}

CUDA_DEVICE void CudaUniformSampler::sample_2d(
    CudaVector<float,2>& output, 
    const CudaVector<float,2>& min_value,
    const CudaVector<float,2>& max_value) {
    output[0] = sample_1d(min_value[0], max_value[0]);
    output[1] = sample_1d(min_value[1], max_value[1]);
}

}
