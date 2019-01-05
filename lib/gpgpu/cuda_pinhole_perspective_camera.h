#pragma once

#include "gpgpu/cuda_camera.h"
#include "scene/camera/pinhole_perspective_camera.h"

namespace cpt {

class CudaPinholePerspectiveCamera : public CudaCamera
{
public:
    CudaPinholePerspectiveCamera(const PinholePerspectiveCamera& camera);

    void generate_rays(CudaRay* rays, size_t width, size_t height) const override;
private:
    PinholePerspectiveCamera _camera;
};

}
