#pragma once

#include "scene/scene.h"
#include <memory>

namespace cpt {

class CudaAccelerationStructure;

class CudaScene
{
public:
    CudaScene(const ScenePtr& scene);
    ~CudaScene();

private: 
    CudaAccelerationStructure* _accel_structure;
};

using CudaScenePtr = std::shared_ptr<CudaScene>;

}
