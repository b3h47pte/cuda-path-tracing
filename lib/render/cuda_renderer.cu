#include "cuda_renderer.h"
#include "gpgpu/cuda_path_integrator.h"
#include "gpgpu/cuda_ptr.h"
#include "gpgpu/cuda_ray.h"
#include "gpgpu/cuda_uniform_sampler.h"
#include "gpgpu/cuda_utils.h"
#include <thrust/copy.h>

#include "gpgpu/cuda_camera.h"
#include <iostream>

namespace cpt {

void CudaRenderer::render(AovOutput& output) const {
    // CudaAovOutput will transfer the contents of AovOutput for use on the GPU.
    // Upon destruction, it will transfer those contents back onto the host.
    CudaAovOutput* cuda_output = cuda_new<CudaAovOutput>(output);
    const size_t film_width = output.width();
    const size_t film_height = output.height();

    // Array of rays to shoot out. Only do one ray per pixel for now (at a time).
    const size_t num_rays = film_width * film_height;
    CudaRay* active_rays = cuda_new_array<CudaRay>(num_rays);
    CudaRay* active_rays_swap = cuda_new_array<CudaRay>(num_rays);
    CudaRay* end_rays = active_rays + num_rays;

    // TODO: Pull from options.
    const size_t samples_per_pixel = 1;
    const size_t max_depth = 1;
    CudaPathIntegrator integrator;
    CudaUniformSampler* samplers = cuda_new_array_device<CudaUniformSampler>(num_rays);

    for (auto spp = 0; spp < samples_per_pixel; ++spp) {
        // Generate an initial set of rays.
        _cuda_scene->generate_rays(samplers, active_rays, film_width, film_height);

        size_t depth = 0;
        while (end_rays != active_rays && depth < max_depth) {
            // Send out rays into scene and do shading computations when necessary.
            integrator.Li(active_rays, end_rays - active_rays, _cuda_scene.get(), spp, cuda_output);

            // Stream compact based on whether or not the ray is dead.
            // Copy active_rays into active_rays_swap. Then swap the pointer.
            end_rays = thrust::copy_if(active_rays, end_rays, active_rays_swap,
            [] CUDA_DEVHOST (const CudaRay& ray){
                return ray.is_alive();
            });

            std::swap(active_rays, active_rays_swap);
            ++depth;
        }
    }
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cuda_output->save(output);

    CHECK_CUDA_ERROR(cudaFree(samplers));
    cuda_delete_array(active_rays, num_rays);
    cuda_delete(cuda_output);
}

}
