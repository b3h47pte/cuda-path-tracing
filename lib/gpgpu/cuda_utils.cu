#include "cuda_utils.h"

namespace cpt {

CUDA_DEVICE int get_cuda_flat_block_index() {
    return blockIdx.z * gridDim.y * gridDim.x +
        blockIdx.y * gridDim.x + blockIdx.x;
}

CUDA_DEVICE int get_cuda_flat_thread_index() {
    return get_cuda_flat_block_index() * blockDim.x * blockDim.y * blockDim.z
        + threadIdx.z * blockDim.y * blockDim.x
        + threadIdx.y * blockDim.x + threadIdx.x;
}

}
