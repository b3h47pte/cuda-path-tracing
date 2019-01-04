#pragma once

#include <cstddef>
#include "gpgpu/cuda_utils.h"
#include "gpgpu/math/cuda_vector.h"
#include <thrust/device_vector.h>

namespace cpt {

class CudaImage
{
public:
    CudaImage(size_t width, size_t height);
    ~CudaImage();

    CUDA_DEVHOST void set_pixel(const CudaVector<float,3>& rgba, size_t x, size_t y);
    CUDA_DEVHOST const CudaVector<float,3>& get_pixel(size_t x, size_t y) const;

private:
    CUDA_DEVHOST size_t index(size_t x, size_t y) const;

    size_t _width;
    size_t _height;

    CudaVector<float,3>* _data;
};

}
