#include "cuda_renderer.h"

namespace cpt {

CudaRenderer::CudaRenderer(const ScenePtr& scene, const std::string& camera_id):
    _cuda_scene(cuda_make_shared<CudaScene>(scene, camera_id)) {
}

}
