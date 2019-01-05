#include "cuda_renderer.h"
#include "gpgpu/cuda_ptr.h"
#include "gpgpu/cuda_ray.h"
#include "gpgpu/cuda_stream_compaction.h"
#include "gpgpu/cuda_utils.h"
#include <thrust/device_vector.h>

namespace cpt {

void CudaRenderer::render(AovOutput& output) const {
    // CudaAovOutput will transfer the contents of AovOutput for use on the GPU.
    // Upon destruction, it will transfer those contents back onto the host.
    CudaAovOutput* cuda_output = cuda_new<CudaAovOutput>(output);

    const size_t num_rays = 100;
    CudaRay* active_rays = cuda_new_array<CudaRay>(num_rays);
    CudaRay* end_rays = active_rays + num_rays;

    // TODO: Pull from options.
    const int samples_per_pixel = 5;
    for (auto spp = 0; spp < samples_per_pixel; ++spp) {
        // Generate an initial set of rays.

        // Send out rays into scene. After they hit something, return and
        // do stream compaction (if enabled) to kill dead rays so we don't
        // do unnecessary work.
        while (end_rays != active_rays) {

            // Stream compact based on whether or not the ray is dead.
            end_rays = cuda_stream_compact(active_rays, end_rays,
            [] CUDA_DEVICE (const CudaRay& ray) {
                return true;
            });
        }

    }
    
    cuda_output->save(output);
    cuda_delete_array(active_rays, num_rays);
    cuda_delete(cuda_output);
}

}
