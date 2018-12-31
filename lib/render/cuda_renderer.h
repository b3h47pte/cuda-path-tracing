#pragma once

#include "gpgpu/cuda_utils.h"
#include "gpgpu/cuda_ptr.h"
#include "gpgpu/cuda_scene.h"
#include <memory>
#include "render/aov_output.h"
#include "scene/scene.h"

namespace cpt {

class CudaRenderer
{
public:
    CudaRenderer(const ScenePtr& scene);

    void render(AovOutput& output) const;
private:
    CudaScenePtr _cuda_scene;
};

using CudaRendererPtr = std::shared_ptr<CudaRenderer>;

}
