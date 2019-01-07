#include "cuda_path_integrator.h"
#include "gpgpu/cuda_acceleration_structure.h"
#include "gpgpu/cuda_intersection.h"
#include "gpgpu/cuda_utils.h"
#include "gpgpu/math/cuda_vector.h"

#define COLOR_INTERSECTION_ERROR 1

namespace cpt {
namespace {

__global__ void path_trace(CudaRay* rays, const CudaScene* scene, size_t sampleIdx, CudaAovOutput* output) {
    const int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t x, y;
    output->get_xy_from_flat_index(x, y, pixelIdx);

    CudaRay& ray = rays[pixelIdx];

    CudaVector<float, 3> rgb;
    // See if we intersect anything in the scene.
    CudaIntersection intersection;
    if (!ray_accel_structure_intersect(&ray, scene->accel_structure(), &intersection)) {
        ray.set_alive(false);
#if COLOR_INTERSECTION_ERROR
        switch (intersection.error) {
        case IntersectionError::StackOutOfBounds:
            rgb[0] = 1.f;
            break;
        default:
            break;
        }
#else
        return;
#endif
    }

#if COLOR_INTERSECTION_ERROR
    if (intersection.hit_geometry) {
#endif

    rgb[intersection.hit_geometry->id() % 3] = 1.f;

#if COLOR_INTERSECTION_ERROR
    }
#endif

    // Output.
    for (size_t imgIdx = 0; imgIdx < output->num_images(); ++imgIdx) {
        AovOutput::Channels channel = output->channel(imgIdx);
        CudaImage* img = output->image(imgIdx);
        if (channel == AovOutput::Channels::FinalImage) {
            img->accumulate(
                rgb / static_cast<float>(sampleIdx + 1),
                x,
                y, 
                static_cast<float>(sampleIdx) / (sampleIdx + 1));
        }
    }
}

}

void CudaPathIntegrator::Li(CudaRay* rays, size_t num_rays, const CudaScene* scene, size_t sampleIdx, CudaAovOutput* output) const {
    int blocks, threads;
    compute_blocks_threads(blocks, threads, num_rays);
    path_trace<<<blocks, threads>>>(rays, scene, sampleIdx, output);
}

}
