#include "cuda_image.h"
#include "gpgpu/cuda_utils.h"
#include "gpgpu/cuda_ptr.h"

namespace cpt {

void CudaImage::accumulate(const CudaVector<float,3>& rgb, size_t x, size_t y, float exist_mul) {
    _data[index(x,y)] = (_data[index(x,y)] * exist_mul) + rgb;
}

void CudaImage::set_pixel(const CudaVector<float,3>& rgb, size_t x, size_t y) {
    _data[index(x,y)] = rgb;
}

const CudaVector<float,3>& CudaImage::get_pixel(size_t x, size_t y) const {
    return _data[index(x,y)];
}

size_t CudaImage::index(size_t x, size_t y) const {
    return y * _width + x;
}

}
