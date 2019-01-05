#include "pinhole_perspective_camera.h"
#include "gpgpu/gpgpu_converter.h"

namespace cpt {

PinholePerspectiveCamera::PinholePerspectiveCamera(
    const Angle& horizontal_fov,
    float film_aspect_ratio,
    const Distance& focal_length,
    const Distance& near_z,
    const Distance& far_z):
    _horizontal_fov(horizontal_fov),
    _film_aspect_ratio(film_aspect_ratio),
    _focal_length(focal_length),
    _near_z(near_z),
    _far_z(far_z)
{
}

void PinholePerspectiveCamera::convert(GpgpuConverter& converter) const {
    converter.convert(*this);
}

}
