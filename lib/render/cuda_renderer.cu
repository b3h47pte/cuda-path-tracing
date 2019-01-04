#include "cuda_renderer.h"
#include "gpgpu/cuda_ptr.h"

#define USE_STREAM_COMPACTION 1

namespace cpt {

void CudaRenderer::render(AovOutput& output) const {
    // CudaAovOutput will transfer the contents of AovOutput for use on the GPU.
    // Upon destruction, it will transfer those contents back onto the host.
    CudaAovOutput* cuda_output = cuda_new<CudaAovOutput>(output);

    // TODO: Pull from options.
    const int samples_per_pixel = 5;
    for (auto spp = 0; spp < samples_per_pixel; ++spp) {
        // Generate an initial set of rays.

        // Send out rays into scene. After they hit something, return and
        // do stream compaction (if enabled) to kill dead rays so we don't
        // do unnecessary work.

    // TODO: Pull from options
#if USE_STREAM_COMPACTION

#endif

    }
    
    cuda_output->save(output);
    cuda_delete(cuda_output);
}



}
