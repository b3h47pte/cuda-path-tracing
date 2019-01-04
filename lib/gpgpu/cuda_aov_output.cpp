#include "cuda_aov_output.h"
#include "gpgpu/cuda_ptr.h"
#include "utilities/error.h"

namespace cpt {

CudaImage* CudaAovOutput::copy_image_to_cuda(AovOutput::Channels channel, const boost::gil::rgb32_image_t& img) {
    CudaImage* new_img = cuda_new<CudaImage>(img.width(), img.height());
    // TODO: Is there a more efficient way to do this? memcpy?
    CudaVector<float,3> rgb;
    for (auto y = 0; y < img.height(); ++y) {
        for (auto x = 0; x < img.width(); ++x) {
            boost::gil::rgb32_pixel_t px = *boost::gil::const_view(img).at(x, y);
            for (int i = 0; i < 3; ++i) {
                rgb[i] = px[i];
            }
            new_img->set_pixel(rgb, x, y);
        }
    }
    return new_img;
}

void CudaAovOutput::copy_image_from_cuda(CudaImage* cuda_image, AovOutput::Channels channel, boost::gil::rgb32_image_t& img) {
    // TODO: Is there a more efficient way to do this? memcpy?
    boost::gil::rgb32_pixel_t px;
    for (auto y = 0; y < img.height(); ++y) {
        for (auto x = 0; x < img.width(); ++x) {
            const auto& rgb = cuda_image->get_pixel(x, y);
            for (int i = 0; i < 3; ++i) {
                px[i] = rgb[i];
            }
            *boost::gil::view(img).at(x, y) = px;
        }
    }
}

}
