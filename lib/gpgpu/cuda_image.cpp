#include "cuda_image.h" 
namespace cpt {

CudaImage::CudaImage(size_t width, size_t height):
    _width(width),
    _height(height) {
    _data = cuda_new_array<CudaVector<float,3>>(_width * _height);
}

CudaImage::~CudaImage() {
    cuda_delete_array<CudaVector<float,3>>(_data, _width * _height);
}

}
