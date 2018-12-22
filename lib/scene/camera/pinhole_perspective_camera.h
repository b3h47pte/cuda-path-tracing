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

private:
    Angle _horizontal_fov;
    float _film_aspect_ratio;
    Distance _focal_length;
    Distance _near_z;
    Distance _far_z;
};

using PinholePerspectiveCameraPtr = std::shared_ptr<PinholePerspectiveCamera>;

}
