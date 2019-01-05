#pragma once

#include "scene/scene.h"
#include <memory>

namespace cpt {

class CudaAccelerationStructure;
class CudaCamera;
class CudaRay;

class CudaScene
{
public:
    CudaScene(const ScenePtr& scene, const std::string& camera_id);
    ~CudaScene();

    const CudaCamera* render_camera() const { return _render_camera; }

    void generate_rays(CudaRay* rays, size_t width, size_t height) const;

private: 
    CudaAccelerationStructure* _accel_structure;
    CudaCamera*                _render_camera;
};

using CudaScenePtr = std::shared_ptr<CudaScene>;

}
