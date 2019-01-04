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

private:
    CudaImage* copy_image_to_cuda(AovOutput::Channels channel, const boost::gil::rgb32_image_t& img);
    void copy_image_from_cuda(CudaImage* cuda_image, AovOutput::Channels channel, boost::gil::rgb32_image_t& img);

    size_t _num_images;
    CudaImage** _cuda_images;
    AovOutput::Channels* _active_channels;
};

}
