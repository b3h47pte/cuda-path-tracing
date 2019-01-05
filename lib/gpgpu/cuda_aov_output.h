#pragma once

#include "gpgpu/cuda_image.h"
#include "gpgpu/cuda_utils.h"
#include "render/aov_output.h"
#include <thrust/device_vector.h>

namespace cpt {

class CudaAovOutput
{
public:
    CUDA_HOST CudaAovOutput(AovOutput& host_output);
    CUDA_HOST ~CudaAovOutput();
    CUDA_HOST void save(AovOutput& host_output);

    CUDA_DEVHOST size_t num_images() const { return _num_images; }
    CUDA_DEVHOST AovOutput::Channels channel(size_t idx) const { return _active_channels[idx]; }
    CUDA_DEVHOST CudaImage* image(size_t idx) const { return _cuda_images[idx]; }
    CUDA_DEVHOST void get_xy_from_flat_index(size_t& x, size_t& y, size_t idx) const;

private:
    CudaImage* copy_image_to_cuda(AovOutput::Channels channel, const boost::gil::rgb32f_image_t& img);
    void copy_image_from_cuda(CudaImage* cuda_image, AovOutput::Channels channel, boost::gil::rgb32f_image_t& img);

    size_t _width;
    size_t _height;
    size_t _num_images;
    CudaImage** _cuda_images;
    AovOutput::Channels* _active_channels;
};

}
