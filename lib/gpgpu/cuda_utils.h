#pragma once

#include "utilities/error.h"

#define CHECK_CUDA_ERROR(x) \
    {auto err = x; \
    if (err != cudaSuccess) { \
        THROW_ERROR("Failed to run CUDA command."); \
    }}

namespace cpt {

void initialize_cuda();

}
