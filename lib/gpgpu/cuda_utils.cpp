#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace cpt {
namespace {

cublasHandle_t cublas_handle;

}

void initialize_cuda() {
    // Ensure that there exists a suitable GPU for execution.
    int device_count = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    CHECK_AND_THROW_ERROR(device_count > 0, "No suitable CUDA Device found.");

    // Use device 0 by default.
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));

    // Initialize CuBlas.
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle));

    std::cout << "===========================================================" << std::endl;
    std::cout << "CUDA Initialization with Device 0" << std::endl;
    std::cout << "\tName: " << prop.name << std::endl;
    std::cout << "\tMemory (GB): " << static_cast<double>(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0 << std::endl;
    std::cout << "\tSM: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "===========================================================" << std::endl;
}

}
