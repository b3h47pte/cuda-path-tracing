#include "cuda_utils.h"

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

void compute_blocks_threads(int& blocks, int& threads, size_t total) {
    constexpr int threads_per_block = 256;
    blocks = (total + threads_per_block - 1) / threads_per_block;
    threads = threads_per_block;
}

void compute_blocks_threads_2d(dim3& blocks, dim3& threads, size_t total_x, size_t total_y) {
    constexpr int threads_per_block_x = 16;
    constexpr int threads_per_block_y = 16;
    blocks = dim3((total_x + threads_per_block_x - 1) / threads_per_block_x,
                  (total_y + threads_per_block_y - 1) / threads_per_block_y);
    threads = dim3(threads_per_block_x, threads_per_block_y);
}

}
