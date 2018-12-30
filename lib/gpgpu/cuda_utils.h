#pragma once

#include <cublas_v2.h>
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

#define CHECK_CUBLAS_ERROR(x) \
    {auto err = x; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        THROW_ERROR("Failed to run CUBLAS command."); \
    }}

namespace cpt {

void initialize_cuda();

}
