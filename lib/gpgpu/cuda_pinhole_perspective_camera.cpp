#include "cuda_pinhole_perspective_camera.h"

namespace cpt {

CudaPinholePerspectiveCamera::CudaPinholePerspectiveCamera(const PinholePerspectiveCamera& camera):
    _horizontal_fov_radians(camera.horizontal_fov().radians()),
    _aspect_ratio(camera.film_aspect_ratio()),
    _focal_length_m(camera.focal_length().meters()),
    _near_z_m(camera.near_z().meters()),
    _far_z_m(camera.far_z().meters()) {
}

}
