#pragma once

#define CHECK_CUDA_ERROR(x) \
    {auto err = x; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(""); \
    }}

namespace cpt {

void initialize_cuda();

}
