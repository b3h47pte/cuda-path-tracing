#include "cuda_renderer.h"
#include "gpgpu/cuda_ptr.h"
#include "gpgpu/cuda_ray.h"
#include "gpgpu/cuda_utils.h"
#include <thrust/device_vector.h>
#include <thrust/remove.h>

namespace cpt {

void CudaRenderer::render(AovOutput& output) const {
    // CudaAovOutput will transfer the contents of AovOutput for use on the GPU.
    // Upon destruction, it will transfer those contents back onto the host.
    CudaAovOutput* cuda_output = cuda_new<CudaAovOutput>(output);

    // TODO: Pull from options.
    const int samples_per_pixel = 5;
    for (auto spp = 0; spp < samples_per_pixel; ++spp) {
        // Generate an initial set of rays.
        thrust::device_vector<CudaRay> active_rays;
        auto end_it = active_rays.end();

        // Send out rays into scene. After they hit something, return and
        // do stream compaction (if enabled) to kill dead rays so we don't
        // do unnecessary work.
        while (end_it != active_rays.begin()) {

            end_it = thrust::remove_if(active_rays.begin(), active_rays.end(), 
            [] CUDA_DEVHOST (const CudaRay& ray) {
                return true;
            });
        }

    }
    
    cuda_output->save(output);
    cuda_delete(cuda_output);
}

}
