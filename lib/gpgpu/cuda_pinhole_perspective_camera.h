#pragma once

#include "gpgpu/cuda_camera.h"
#include "scene/camera/pinhole_perspective_camera.h"
#include "gpgpu/cuda_utils.h"

namespace cpt {

class CudaPinholePerspectiveCamera : public CudaCamera
{
public:
    CudaPinholePerspectiveCamera(const PinholePerspectiveCamera& camera);

    void generate_rays(CudaSampler* samplers, CudaRay* rays, size_t width, size_t height) const override;

    CUDA_DEVHOST float horizontal_fov_radians() const { return _horizontal_fov_radians; }
    CUDA_DEVHOST float aspect_ratio() const { return _aspect_ratio; }
    CUDA_DEVHOST float focal_length_m() const { return _focal_length_m; }
    CUDA_DEVHOST float near_z_m() const { return _near_z_m; }
    CUDA_DEVHOST float far_z_m() const { return _far_z_m; }

private:
    float _horizontal_fov_radians;
    float _aspect_ratio;
    float _focal_length_m;
    float _near_z_m;
    float _far_z_m;
};

}
