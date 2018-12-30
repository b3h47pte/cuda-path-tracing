#include "cuda_scene.h"
#include "gpgpu/cuda_bvh.h"
#include "gpgpu/cuda_utils.h"

namespace cpt {

CudaScene::CudaScene(const ScenePtr& scene) {
    _accel_structure = cuda_new<CudaBVH>(scene->cuda_geometry(true));
}

CudaScene::~CudaScene() {
    cuda_delete(_accel_structure);
}

}
