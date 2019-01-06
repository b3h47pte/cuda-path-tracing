#pragma once

#include "scene/camera/camera.h"

#include "math/angles.h"
#include "math/distance.h"

namespace cpt {

class PinholePerspectiveCamera: public Camera
{
public:
    PinholePerspectiveCamera(
        const Angle& horizontal_fov,
        float film_aspect_ratio,
        const Distance& focal_length,
        const Distance& near_z,
        const Distance& far_z);

    const Angle& horizontal_fov() const { return _horizontal_fov; }
    float film_aspect_ratio() const { return _film_aspect_ratio; }
    const Distance& focal_length() const { return _focal_length; }
    const Distance& near_z() const { return _near_z; }
    const Distance& far_z() const { return _far_z; }

    void convert(GpgpuConverter& converter) const override;

protected:
    Angle _horizontal_fov;
    float _film_aspect_ratio;
    Distance _focal_length;
    Distance _near_z;
    Distance _far_z;
};

using PinholePerspectiveCameraPtr = std::shared_ptr<PinholePerspectiveCamera>;

}
