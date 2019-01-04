#include "cuda_renderer.h"

namespace cpt {

CudaRenderer::CudaRenderer(const ScenePtr& scene):
    _cuda_scene(cuda_make_shared<CudaScene>(scene)) {
}

}
