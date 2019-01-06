#pragma once

#include "scene/scene.h"
#include <memory>
#include "gpgpu/cuda_utils.h"

namespace cpt {

class CudaAccelerationStructure;
class CudaCamera;
class CudaRay;
class CudaSampler;

class CudaScene
{
public:
    CudaScene(const ScenePtr& scene, const std::string& camera_id);
    ~CudaScene();

    CUDA_DEVHOST const CudaCamera* render_camera() const { return _render_camera; }
    CUDA_DEVHOST const CudaAccelerationStructure* accel_structure() const { return _accel_structure; }

    void generate_rays(CudaSampler* samplers, CudaRay* rays, size_t width, size_t height) const;

private: 
    CudaAccelerationStructure* _accel_structure;
    CudaCamera*                _render_camera;
};

using CudaScenePtr = std::shared_ptr<CudaScene>;

}
