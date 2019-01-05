#include "cuda_aov_output.h"

namespace cpt {

CudaAovOutput::CudaAovOutput(AovOutput& host_output):
    _width(host_output.width()),
    _height(host_output.height()) {
    auto active_channels = host_output.active_channels();
    _num_images = active_channels.size();

    CHECK_CUDA_ERROR(cudaMallocManaged(&_cuda_images, sizeof(CudaImage*) * _num_images));
    CHECK_CUDA_ERROR(cudaMallocManaged(&_active_channels, sizeof(AovOutput::Channels) * _num_images));

    for (auto i = 0; i < active_channels.size(); ++i) {
        const AovOutput::Channels channel = active_channels[i];
        CudaImage* new_cuda_img = copy_image_to_cuda(channel, host_output.image(channel));

        _active_channels[i] = channel;
        _cuda_images[i] = new_cuda_img;
    }
}

CudaAovOutput::~CudaAovOutput() {
    for (size_t i = 0; i < _num_images; ++i) {
        cuda_delete(_cuda_images[i]);
    }
    cudaFree(_cuda_images);
    cudaFree(_active_channels);
}

void CudaAovOutput::save(AovOutput& host_output) {
    for (auto i = 0; i < _num_images; ++i) {
        const AovOutput::Channels channel = _active_channels[i];
        copy_image_from_cuda(_cuda_images[i], channel, host_output.image(channel));
    }
}

void CudaAovOutput::get_xy_from_flat_index(size_t& x, size_t& y, size_t idx) const {
    x = idx % _width;
    y = idx / _width;
}

}
