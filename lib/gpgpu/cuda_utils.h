#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "utilities/error.h"
#include <iostream>

#ifdef __CUDACC__
#define CUDA_HOST __host__
#define CUDA_DEVHOST __host__ __device__
#else
#define CUDA_HOST
#define CUDA_DEVHOST
#endif

#ifndef NO_CUDA_CHECKS
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
#else
#define CHECK_CUDA_ERROR(x) x
#define CHECK_CUBLAS_ERROR(x) x
#endif

namespace cpt {

void initialize_cuda();

}
