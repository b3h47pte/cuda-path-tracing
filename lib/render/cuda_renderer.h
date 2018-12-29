#pragma once

#include "gpgpu/cuda_utils.h"
#include <memory>
#include "scene/scene.h"

namespace cpt {

class CudaRenderer
{
public:
    CudaRenderer(const ScenePtr& scene);

    void render() const;
private:
};

using CudaRendererPtr = std::shared_ptr<CudaRenderer>;

}
