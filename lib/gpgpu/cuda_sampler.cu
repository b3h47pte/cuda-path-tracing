#include "cuda_sampler.h"
#include "gpgpu/cuda_utils.h"

namespace cpt {

CUDA_DEVICE CudaSampler::CudaSampler(unsigned long long seed) {
    generate_curand_state(seed); 
}

CUDA_DEVICE CudaSampler::~CudaSampler() {
}

CUDA_DEVICE void CudaSampler::generate_curand_state(unsigned long long seed) {
    curand_init(seed, get_cuda_flat_thread_index(), 0, &_curand_state);
}

}
