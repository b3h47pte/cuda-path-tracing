#pragma once

#include "utilities/error.h"

#ifdef __CUCDACC__
#define CUDA_HOST __host__
#else
#define CUDA_HOST
#endif

#define CHECK_CUDA_ERROR(x) \
    {auto err = x; \
    if (err != cudaSuccess) { \
        THROW_ERROR("Failed to run CUDA command."); \
    }}

namespace cpt {

void initialize_cuda();

}
